import os
import re
import smtplib
import numpy as np
import logging
from urllib.parse import urlparse, parse_qs
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from sqlalchemy import create_engine, text
from sshtunnel import SSHTunnelForwarder
from superagi.tools.base_tool import BaseTool, ToolConfiguration
from pydantic import BaseModel, Field
from typing import Type, Optional
from email.message import EmailMessage
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict

# Включение логирования действий агента
logging.basicConfig(filename="agent_actions.log", level=logging.INFO)

TEST_MODE = True  # Включение тестового режима

class GoogleAdsOptimizerInput(BaseModel):
    campaign_id: str = Field(..., description="ID рекламной кампании для оптимизации.")
    max_cpa: float = Field(..., description="Максимальная допустимая стоимость за конверсию.")
    min_conversion_rate: float = Field(..., description="Минимальная допустимая конверсия (например, 0.05 для 5%).")
    attribution_window_days: int = Field(30, description="Количество дней для учета задержек в конверсии.")
    max_budget: float = Field(..., description="Максимальный бюджет кампании.")
    daily_budget_limit: float = Field(..., description="Максимальный дневной бюджет.")
    optimization_strategy: str = Field("ROAS", description="Стратегия оптимизации: 'ROAS', 'CPA', 'Manual'.")

class GoogleAdsOptimizer(BaseTool):
    name: str = "Google Ads Optimizer"
    args_schema: Type[BaseModel] = GoogleAdsOptimizerInput
    description: str = "Оптимизация Google Ads кампаний на основе данных о продажах и машинного обучения."

    def _initialize_google_ads_client(self):
        """Инициализация клиента Google Ads с использованием конфигурации инструмента."""
        config = {
            "developer_token": self.get_tool_config("GOOGLE_ADS_DEVELOPER_TOKEN"),
            "client_id": self.get_tool_config("GOOGLE_ADS_CLIENT_ID"),
            "client_secret": self.get_tool_config("GOOGLE_ADS_CLIENT_SECRET"),
            "refresh_token": self.get_tool_config("GOOGLE_ADS_REFRESH_TOKEN"),
            "login_customer_id": self.get_tool_config("GOOGLE_ADS_LOGIN_CUSTOMER_ID"),
            "token_uri": "https://oauth2.googleapis.com/token",
            "use_proto_plus": True
        }
        missing_keys = [key for key, value in config.items() if not value]
        if missing_keys:
            logging.error(f"❌ Отсутствуют обязательные параметры: {missing_keys}")
            raise ValueError(f"❌ Отсутствуют обязательные параметры: {missing_keys}")
        return GoogleAdsClient.load_from_dict(config)

    def _apply_optimization_strategy(self, keyword_data, strategy, max_cpa, min_conversion_rate):
        """Применяет стратегию оптимизации ставок в зависимости от выбранного метода."""
        suggested_changes = {}
        for keyword, stats in keyword_data.items():
            avg_cpa = stats["total_sales"] / stats["conversion_count"] if stats["conversion_count"] > 0 else float('inf')
            conv_rate = stats["conversion_count"] / stats["clicks"] if stats["clicks"] > 0 else 0.0
            if strategy == "ROAS" and stats["total_sales"] > 0:
                suggested_changes[keyword] = "Increase bid"
            elif strategy == "CPA" and avg_cpa > max_cpa:
                suggested_changes[keyword] = "Decrease bid"
            elif strategy == "Manual":
                suggested_changes[keyword] = "Review manually"
        return suggested_changes  

    def _fetch_sales_data(self, attribution_window_days: int):
        """Функция загрузки данных о продажах из базы данных."""
        database_url = self.get_tool_config("DATABASE_URL")
        engine = create_engine(database_url)
        query = text(f"""
            SELECT * FROM clicks
            WHERE data >= NOW() - INTERVAL {attribution_window_days} DAY
            AND otkudaAds = 'y'
            AND kuda LIKE '%%gclid%%'
            AND cost > 0
        """)
        with engine.connect() as connection:
            result = connection.execute(query)
            sales_data = result.fetchall()
        return sales_data

    def _map_gclid_to_keyword(self, google_ads_client, gclid_list):
        gclid_to_keyword = {}
        try:
            service = google_ads_client.get_service("GoogleAdsService")
            for gclid in gclid_list:
                query = f"""
                    SELECT click_view.gclid, ad_group_criterion.keyword.text
                    FROM click_view
                    WHERE click_view.gclid = '{gclid}'
                """
                response = service.search_stream(customer_id=self.get_tool_config("GOOGLE_ADS_LOGIN_CUSTOMER_ID"), query=query)
                for batch in response:
                    for row in batch.results:
                        gclid_value = row.click_view.gclid
                        keyword_text = row.ad_group_criterion.keyword.text
                        gclid_to_keyword[gclid_value] = keyword_text
        except GoogleAdsException as ex:
            logging.error(f"Ошибка Google Ads API: {ex}")
        return gclid_to_keyword

    def _calculate_sales_per_keyword(self, sales_data, gclid_map):
        keyword_data = defaultdict(lambda: {"total_sales": 0.0, "conversion_count": 0, "clicks": 0})
        for row in sales_data:
            parsed_url = urlparse(row.kuda)
            query_params = parse_qs(parsed_url.query)
            gclid = query_params.get("gclid", [None])[0]
            keyword = gclid_map.get(gclid, gclid or "unknown")
            keyword_data[keyword]["clicks"] += 1
            keyword_data[keyword]["total_sales"] += float(row.cost) if row.cost else 0.0
            if row.conv in ["registr", "transfer"]:
                keyword_data[keyword]["conversion_count"] += 1
        return keyword_data
        
   def _execute(self, campaign_id: str, max_cpa: float, min_conversion_rate: float, 
                  attribution_window_days: int, max_budget: float, daily_budget_limit: float, optimization_strategy: str):
        logging.info(f"🔹 Запуск оптимизации кампании {campaign_id} в тестовом режиме: {TEST_MODE}")
        google_ads_client = self._initialize_google_ads_client()
        sales_data = self._fetch_sales_data(attribution_window_days)
        gclid_list = [parse_qs(urlparse(row.kuda).query).get("gclid", [None])[0] for row in sales_data]
        gclid_map = self._map_gclid_to_keyword(google_ads_client, gclid_list)
        keyword_data = self._calculate_sales_per_keyword(sales_data, gclid_map)

        detailed_report = {
            keyword: {
                "total_sales": data["total_sales"],
                "conversion_count": data["conversion_count"],
                "average_sale": round(data["total_sales"] / data["conversion_count"], 2) if data["conversion_count"] > 0 else 0.0,
                "status": "effective" if data["conversion_count"] > 0 or data["total_sales"] > 0 else "ineffective"
            } for keyword, data in keyword_data.items()
        }

        optimization_result = {
            "campaign_id": campaign_id,
            "strategy": optimization_strategy,
            "suggested_changes": self._apply_optimization_strategy(keyword_data, optimization_strategy, max_cpa, min_conversion_rate),
            "conversion_report": detailed_report
        }

        if TEST_MODE:
            print("🛑 Агент работает в тестовом режиме! Изменения не применяются.")
            print("🔹 Подробный отчет:", optimization_result)
        else:
            self._apply_google_ads_changes(optimization_result)

        self._save_report_to_file("optimization_report.txt", optimization_result)
        return optimization_result       

    def _save_report_to_file(self, filename, report_content):
        with open(filename, "w") as file:
            file.write(str(report_content))
        logging.info(f"📄 Отчет сохранен в файл {filename}")
