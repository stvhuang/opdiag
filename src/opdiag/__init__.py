from abc import ABC, abstractmethod
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


class Contract(ABC):
    def __init__(
        self,
        strike: float,
        premium: float,
        side: Literal["L", "S"],
    ) -> None:
        if strike <= 0.0 or premium < 0.0 or side not in {"L", "S"}:
            raise ValueError

        self._strike = strike
        self._premium = premium
        self._side = side

    def plot(
        self,
        underlying_price_min: float = 0.0,
        underlying_price_max: float = 200.0,
    ) -> None:
        CompositeContract([self]).plot(underlying_price_min, underlying_price_max)

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def premium(self) -> float:
        return self._premium

    @property
    def side(self) -> Literal["L", "S"]:
        return self._side

    @abstractmethod
    def _pnl(
        self,
        underlying_prices: np.ndarray,
    ) -> np.ndarray:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(strike={self._strike}, premium={self._premium}, side='{self._side}')"

    def __add__(self, rhs: "Contract | CompositeContract") -> "CompositeContract":
        if isinstance(rhs, Contract):
            return CompositeContract([self, rhs])

        if isinstance(rhs, CompositeContract):
            return CompositeContract([self] + rhs._contracts)

        raise TypeError

    def __mul__(self, qty: int) -> "CompositeContract":
        if not isinstance(qty, int) or qty <= 0:
            raise ValueError

        return CompositeContract([self] * qty)


class Call(Contract):
    def _pnl(
        self,
        underlying_prices: np.ndarray,
    ) -> np.ndarray:
        intrinsic_value = np.maximum(underlying_prices - self._strike, 0)

        match self.side:
            case "L":
                return intrinsic_value - self._premium

            case "S":
                return self._premium - intrinsic_value

            case _:
                raise ValueError


class Put(Contract):
    def _pnl(
        self,
        underlying_prices: np.ndarray,
    ) -> np.ndarray:
        intrinsic_value = np.maximum(self._strike - underlying_prices, 0)

        match self.side:
            case "L":
                return intrinsic_value - self._premium

            case "S":
                return self._premium - intrinsic_value

            case _:
                raise ValueError


class CompositeContract:
    def __init__(self, contracts: list[Contract]) -> None:
        self._contracts = contracts

    def _pnl(
        self,
        underlying_prices: np.ndarray,
    ) -> np.ndarray:
        return np.sum(
            [contract._pnl(underlying_prices) for contract in self._contracts], axis=0
        )

    def plot(
        self,
        underlying_price_min: float = 0,
        underlying_price_max: float = 200,
    ) -> None:
        underlying_prices = np.linspace(
            underlying_price_min,
            underlying_price_max,
            int(underlying_price_max - underlying_price_min) + 1,
        )
        pnl = self._pnl(underlying_prices)

        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(
            underlying_prices,
            pnl,
            color="blue",
            label="PnL",
        )
        plt.fill_between(
            underlying_prices, pnl, 0, where=(pnl > 0), color="g", alpha=0.2
        )
        plt.fill_between(
            underlying_prices, pnl, 0, where=(pnl < 0), color="r", alpha=0.2
        )
        plt.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=2,
        )
        for contract in self._contracts:
            plt.axvline(
                contract.strike,
                alpha=0.5,
                color="red",
                linestyle="--",
            )
        plt.legend()

        plt.xlabel("Underlying Price at Expiration")
        plt.ylabel("PnL")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"CompositeContract({self._contracts})"


__all__ = [
    "Contract",
    "Call",
    "Put",
    "CompositeContract",
]


def __dir__():
    return __all__
