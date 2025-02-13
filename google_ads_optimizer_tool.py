import os
import re
import smtplib
import numpy as np
import logging
from urllib.parse import urlparse, parse_qs
from google.ads.googleads.client import GoogleAdsClient
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

    def _fetch_sales_data(self, attribution_window_days: int):
        """Функция загрузки данных о продажах из базы данных."""
        database_url = self.get_tool_config("DATABASE_URL")
        engine = create_engine(database_url)
        query = text(f"""
            SELECT * FROM clicks
            WHERE data >= NOW() - INTERVAL {attribution_window_days} DAY
            AND otkudaAds = 'y' AND kuda LIKE '%%gbraid%%'
        """)
        with engine.connect() as connection:
            result = connection.execute(query)
            sales_data = result.fetchall()
        return sales_data

    def _calculate_sales_per_ad(self, sales_data):
        """Анализирует продажи по объявлениям, используя gbraid."""
        ad_sales = defaultdict(lambda: {"total_sales": 0.0, "conversion_count": 0})
        
        for row in sales_data:
            kuda = row.kuda  # URL страницы перехода
            cost = float(row.cost) if row.cost is not None else 0.0  # Преобразуем cost в float
            conv = row.conv  # Тип конверсии (registr или transfer)

            # Извлекаем gbraid из URL
            parsed_url = urlparse(kuda)
            query_params = parse_qs(parsed_url.query)
            gbraid = query_params.get("gbraid", [None])[0]  # Берем первый gbraid, если есть

            if gbraid:
                ad_sales[gbraid]["total_sales"] += cost
                if conv in ["registr", "transfer"]:
                    ad_sales[gbraid]["conversion_count"] += 1

        return ad_sales

    def _execute(self, campaign_id: str, max_cpa: float, min_conversion_rate: float, 
                  attribution_window_days: int, max_budget: float, daily_budget_limit: float, optimization_strategy: str):
        logging.info(f"🔹 Запуск оптимизации кампании {campaign_id} в тестовом режиме: {TEST_MODE}")
        google_ads_client = self._initialize_google_ads_client()
        sales_data = self._fetch_sales_data(attribution_window_days)
        sales_per_ad = self._calculate_sales_per_ad(sales_data)

        # Логирование результатов
        logging.info(f"🔹 Анализ продаж по объявлениям: {sales_per_ad}")
        print("🔹 Анализ продаж по объявлениям:", sales_per_ad)
        
        optimization_result = {
            "suggested_changes": self._apply_optimization_strategy(campaign_id, optimization_strategy, max_cpa, min_conversion_rate)
        }
        
        if TEST_MODE:
            print("🛑 Агент работает в тестовом режиме! Изменения не применяются.")
            print("🔹 Предлагаемые изменения:", optimization_result)
        else:
            self._apply_google_ads_changes(optimization_result)
        
        self._save_report_to_file("optimization_report.txt", optimization_result)
        return optimization_result

    def _save_report_to_file(self, filename, report_content):
        with open(filename, "w") as file:
            file.write(str(report_content))
        logging.info(f"📄 Отчет сохранен в файл {filename}")
