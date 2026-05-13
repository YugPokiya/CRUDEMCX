# MCX Data Transformation Quick Notes

## Medallion flow
1. Bronze: raw import + metadata
2. Silver: type cleanup + dedupe + percent-change columns
3. Gold: technical indicators + next-day class labels

## Label mapping
- `loss` -> 0
- `neutral` -> 1
- `profit` -> 2

## Trading interpretation in backtest
- Predicted `profit`: long next-day return
- Predicted `loss`: short next-day return
- Predicted `neutral`: no trade
