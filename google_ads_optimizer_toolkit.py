from abc import ABC
import os
from superagi.tools.base_tool import BaseToolkit, BaseTool, ToolConfiguration
from typing import List
from google_ads_optimizer_tool import GoogleAdsOptimizer
from superagi.types.key_type import ToolConfigKeyType

class GoogleAdsOptimizerToolkit(BaseToolkit, ABC):
    name: str = "Google Ads Optimizer Toolkit"
    description: str = "Toolkit for optimizing Google Ads campaigns based on sales data."

    def get_tools(self) -> List[BaseTool]:
        return [GoogleAdsOptimizer()]

    def get_env_keys(self) -> List[ToolConfiguration]:
        return [
            ToolConfiguration(key="GOOGLE_ADS_CLIENT_ID", key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=True),
            ToolConfiguration(key="GOOGLE_ADS_CLIENT_SECRET", key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=True),
            ToolConfiguration(key="GOOGLE_ADS_REFRESH_TOKEN", key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=True),
            ToolConfiguration(key="GOOGLE_ADS_DEVELOPER_TOKEN", key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=True),
            ToolConfiguration(key="GOOGLE_ADS_LOGIN_CUSTOMER_ID", key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=True),
            ToolConfiguration(key="DATABASE_URL", key_type=ToolConfigKeyType.STRING, is_required=False, is_secret=True)
        ]
