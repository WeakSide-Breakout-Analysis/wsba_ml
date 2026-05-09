## WSBA ML

Collection of modelling efforts utilized in the production of analytics-based hockey content shared online.  Cloning and running the repository, given that the input play-by-play data is obtained through the wsba_hockey Python package, should mostly reproduce this content.  Some files are ommitted due to Git's file size limits.

### Models

- Expected Goals (WSBA-xG): Basic XGBoost goal prediction model.  Two models are built: one for situations goaltenders are in the net and situations goaltenders are out of the net (empty net).  Developed on qualified unblocked shot attempts from 2021-22 to 2024-25.  The NHL has included special shot types ('short' and 'behind-net') which are incorporated in the model, but unused for goaltender statistics.  Expected goal values are the basis for most of all other current and future models.  Performance metrics on this model are included in the repository.

- Regularized Adjusted Plus-Minus (WSBA-RAPM): Ridge regression models developed to "isolate" contextual factors for individual contributions to team outcomes, using on-ice shot rates, penalty rates, and individual unblocked shot attempts as targets for the regressions.  Model is ran for each season.  Coefficients, including the shot map coefficients for each team, are included in the repository.

- Goals Above Replacement (WSBA-GAR): Implementation of RAPM models expanded by a player's observed time on ice for their "total" value, using an adjustment from the league-average to express the value of a player's individual contributions in terms of goals, above a replacement level NHL (which are somewhat below average skaters or goaltenders).  The file "qoc_qot.py" estimates each player's quality of competition and teammates based on the generated GAR values, using the time on ice spent against or with a certain skater as the weight; combining these together derives the desired values.

- Game Score (WSBA-GS): A basic, primitive Random Forest model primarily designed to describe single-game performances by skater and goaltender in NHL games.  Model is trained on regular season games from 2021-22 to 2024-25.  Model outputs are largely used to estimate a player's game-to-game consistency.  The best player performance estimates are from the RAPM and ensuing GAR models.

- Game Simulation Engine (WSBA-GSE): A simulation engine utilizing both individual performance estimates (from the RAPM models) and observed rates in conjunction with league-based constants to simulate the outcomes of NHL games.  This model is primarily utilized to predict the likeliest sets of outcomes for NHL games, including the victor, team outputs, and individual player box-score statistics.
