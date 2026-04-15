from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    anthropic_api_key: str = Field(default="")
    polygon_api_key: str = Field(default="")

    # IBKR
    ibkr_host: str = Field(default="127.0.0.1")
    ibkr_port: int = Field(default=7497)
    ibkr_client_id: int = Field(default=1)

    # Trading mode
    trading_mode: str = Field(default="paper")  # "paper" or "live"

    # Risk
    max_risk_per_trade_pct: float = Field(default=1.0)
    max_daily_loss_pct: float = Field(default=6.0)
    max_concurrent_positions: int = Field(default=3)

    # Paper account
    paper_account_size: float = Field(default=100_000.0)

    # Database
    db_path: str = Field(default="./trading_bot.duckdb")

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"


settings = Settings()
