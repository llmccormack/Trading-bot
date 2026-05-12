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
    topstep_account_size:     float = Field(default=50_000.0) # combine account size ($50K plan)
    topstep_daily_loss_limit: float = Field(default=1_000.0)  # hard daily stop-out (buffer below $2k trailing limit)
    topstep_profit_target:    float = Field(default=3_000.0)  # $50K account target (was wrongly set to $6k)
    topstep_max_contracts:    int   = Field(default=5)        # TopStep $50K = 5 mini / 50 micro hard cap
    topstep_min_trade_days:   int   = Field(default=10)       # min calendar days before profit target counts

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"

    @property
    def is_tradovate(self) -> bool:
        return self.trading_mode.startswith("tradovate")


settings = Settings()
