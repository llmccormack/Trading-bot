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

    # Trading mode — "paper" | "tradovate_demo" | "tradovate_live"
    trading_mode: str = Field(default="paper")

    # Tradovate credentials (only needed when trading_mode starts with "tradovate")
    tradovate_username:   str = Field(default="")
    tradovate_password:   str = Field(default="")
    tradovate_app_id:     str = Field(default="")
    tradovate_app_secret: str = Field(default="")
    tradovate_cid:        str = Field(default="")
    tradovate_sec:        str = Field(default="")
    tradovate_is_demo:    str = Field(default="true")   # "true" = demo, "false" = live

    # Risk
    max_risk_per_trade_pct: float = Field(default=1.0)
    max_daily_loss_pct: float = Field(default=6.0)
    max_concurrent_positions: int = Field(default=3)

    # Paper account
    paper_account_size: float = Field(default=100_000.0)

    # Database
    db_path: str = Field(default="./trading_bot.duckdb")

    # TopStep combine mode
    # When enabled: tighter score thresholds, hard daily loss cap, no shorts, lower VIX block
    topstep_mode: bool = Field(default=False)
    topstep_daily_loss_limit: float = Field(default=1000.0)   # hard $ stop-out for the combine
    topstep_profit_target:    float = Field(default=6000.0)   # $100k account target

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"

    @property
    def is_tradovate(self) -> bool:
        return self.trading_mode.startswith("tradovate")


settings = Settings()
