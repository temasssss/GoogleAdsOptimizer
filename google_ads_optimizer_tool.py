import os
import re
import smtplib
import numpy as np
import logging
from urllib.parse import urlparse, parse_qs
from google.ads.googleads.client import GoogleAdsClient
from sqlalchemy import create_engine, text
from sshtunnel import SSHTunnelForwarder
from superagi.tools.base_tool import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from email.message import EmailMessage
from sklearn.ensemble import RandomForestRegressor

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
        """Инициализация клиента Google Ads."""
        return GoogleAdsClient.load_from_storage("google-ads.yaml")

    def _execute(self, campaign_id: str, max_cpa: float, min_conversion_rate: float, 
                  attribution_window_days: int, max_budget: float, daily_budget_limit: float, optimization_strategy: str):
        logging.info(f"🔹 Запуск оптимизации кампании {campaign_id} в тестовом режиме: {TEST_MODE}")
        google_ads_client = self._initialize_google_ads_client()
        sales_data = self._fetch_sales_data(attribution_window_days)
        model = self._train_bid_model(sales_data)
        self._check_budget_limits(google_ads_client, campaign_id, daily_budget_limit, max_budget)
        self._analyze_keywords(google_ads_client, campaign_id)
        
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

    def _apply_optimization_strategy(self, campaign_id, strategy, cpa, roas):
        if strategy == "ROAS":
            if roas > 3:
                return "Увеличение ставок, ROAS высокий"
            elif roas < 1:
                return "Снижение ставок, ROAS низкий"
        elif strategy == "CPA":
            if cpa > self.max_cpa:
                return "Снижение ставок, CPA выше допустимого"
            elif cpa < self.max_cpa * 0.8:
                return "Увеличение ставок, CPA низкий"
        return "Ставки без изменений"

    def _safe_bid_adjustment(self, new_bid, current_bid):
        max_change = 0.3
        if new_bid > current_bid * (1 + max_change):
            new_bid = current_bid * (1 + max_change)
        elif new_bid < current_bid * (1 - max_change):
            new_bid = current_bid * (1 - max_change)
        return new_bid

    def _log_action(self, action, details):
        log_message = f"🔹 Действие: {action} | Детали: {details}"
        logging.info(log_message)
        print(log_message)

    def _save_report_to_file(self, filename, report_content):
        with open(filename, "w") as file:
            file.write(str(report_content))
        logging.info(f"📄 Отчет сохранен в файл {filename}")
