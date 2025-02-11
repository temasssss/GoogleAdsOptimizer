import os
import re
from urllib.parse import urlparse, parse_qs
from google.ads.googleads.client import GoogleAdsClient
from sqlalchemy import create_engine, text
from sshtunnel import SSHTunnelForwarder
from superagi.tools.base_tool import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional


class GoogleAdsOptimizerInput(BaseModel):
    campaign_id: str = Field(..., description="ID рекламной кампании для оптимизации.")
    max_cpa: float = Field(..., description="Максимальная допустимая стоимость за конверсию.")
    min_conversion_rate: float = Field(..., description="Минимальная допустимая конверсия (например, 0.05 для 5%).")
    attribution_window_days: int = Field(30, description="Количество дней для учета задержек в конверсии.")


class GoogleAdsOptimizer(BaseTool):
    name: str = "Google Ads Optimizer"
    args_schema: Type[BaseModel] = GoogleAdsOptimizerInput
    description: str = "Оптимизация Google Ads кампаний на основе данных о продажах."

    def _execute(self, campaign_id: str, max_cpa: float, min_conversion_rate: float, attribution_window_days: int):
        # Инициализация клиента Google Ads
        google_ads_client = self._initialize_google_ads_client()

        # Подключение к базе данных MySQL и получение данных о продажах
        sales_data = self._fetch_sales_data(attribution_window_days)

        # Оптимизация кампании на основе данных о продажах
        optimization_result = self._optimize_campaign(
            google_ads_client, campaign_id, sales_data, max_cpa, min_conversion_rate
        )

        return optimization_result

    def _initialize_google_ads_client(self):
        credentials = {
            "developer_token": os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
            "client_id": os.getenv("GOOGLE_ADS_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_ADS_CLIENT_SECRET"),
            "refresh_token": os.getenv("GOOGLE_ADS_REFRESH_TOKEN"),
            "login_customer_id": os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
        }

        # Сохранение конфигурации в файл
        with open("google-ads.yaml", "w") as f:
            f.write(f"""
developer_token: {credentials['developer_token']}
client_id: {credentials['client_id']}
client_secret: {credentials['client_secret']}
refresh_token: {credentials['refresh_token']}
login_customer_id: {credentials['login_customer_id']}
""")

        return GoogleAdsClient.load_from_storage("google-ads.yaml")

    def _fetch_sales_data(self, attribution_window_days: int):
        ssh_host = os.getenv("SSH_HOST")
        ssh_port = int(os.getenv("SSH_PORT", 22))
        ssh_username = os.getenv("SSH_USERNAME")
        ssh_password = os.getenv("SSH_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = int(os.getenv("DB_PORT", 3306))
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")

        with SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_username,
            ssh_password=ssh_password,
            remote_bind_address=(db_host, db_port)
        ) as tunnel:
            local_port = tunnel.local_bind_port
            database_url = f"mysql+pymysql://{db_user}:{db_password}@127.0.0.1:{local_port}/{db_name}"

            engine = create_engine(database_url)
            query = text(f"""
            SELECT id, data, comp, otkuda, otkudaAds, kuda, ip, ref, conv, cost, fromCountry, toCountry, fromCur, toCur, amount, lang
            FROM clicks
            WHERE data >= NOW() - INTERVAL {attribution_window_days} DAY
            AND otkudaAds = 'y'
            """)

            with engine.connect() as connection:
                result = connection.execute(query)
                sales_data = result.fetchall()

        return sales_data

    def _extract_gbraid(self, url: str) -> Optional[str]:
        """
        Извлечение параметра gbraid из URL.
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get('gbraid', [None])[0]

    def _optimize_campaign(self, client, campaign_id, sales_data, max_cpa, min_conversion_rate):
        ga_service = client.get_service("GoogleAdsService")

        optimization_log = ""
        for sale in sales_data:
            if sale.otkudaAds == 'y':
                gbraid = self._extract_gbraid(sale.kuda)
                if gbraid:  # Проверка, что переход из Google Ads
                    query = f"""
                    SELECT
                        ad_group_criterion.criterion_id,
                        metrics.conversions,
                        metrics.cost_micros,
                        metrics.conversions_value,
                        ad_group_criterion.cpc_bid_micros
                    FROM ad_group_criterion
                    WHERE campaign.id = {campaign_id}
                    AND segments.gbraid = '{gbraid}'
                    """

                    response = ga_service.search_stream(customer_id=os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"), query=query)

                    for batch in response:
                        for row in batch.results:
                            criterion_id = row.ad_group_criterion.criterion_id
                            conversions = row.metrics.conversions
                            cost = row.metrics.cost_micros / 1_000_000
                            cpa = cost / conversions if conversions else float('inf')
                            current_bid = row.ad_group_criterion.cpc_bid_micros / 1_000_000

                            if conversions < min_conversion_rate or cpa > max_cpa:
                                new_bid = current_bid * 0.9  # Уменьшение ставки на 10%
                                action = "уменьшена"
                            else:
                                new_bid = current_bid * 1.1  # Увеличение ставки на 10%
                                action = "увеличена"

                            self._adjust_bid(client, campaign_id, criterion_id, new_bid)
                            optimization_log += f"Ставка для ключевого слова {criterion_id} {action} до {new_bid:.2f} (CPA={cpa:.2f}, Конверсии={conversions})\n"

        return optimization_log

    def _adjust_bid(self, client, campaign_id, criterion_id, new_bid):
        ad_group_criterion_service = client.get_service("AdGroupCriterionService")
        ad_group_criterion_operation = client.get_type("AdGroupCriterionOperation")
        
        ad_group_criterion = ad_group_criterion_operation.update
        ad_group_criterion.resource_name = ad_group_criterion_service.ad_group_criterion_path(
            os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"), campaign_id, criterion_id
        )
        ad_group_criterion.cpc_bid_micros = int(new_bid * 1_000_000)

        field_mask = client.get_type("FieldMask")
        field_mask.paths.append("cpc_bid_micros")
        ad_group_criterion_operation.update_mask.CopyFrom(field_mask)

        ad_group_criterion_service.mutate_ad_group_criteria(
            customer_id=os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"),
            operations=[ad_group_criterion_operation]
        )
