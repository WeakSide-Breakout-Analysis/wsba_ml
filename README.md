## WSBA ML

[WeakSide Breakout Analysis Web Application](weaksidebreakout.com)

Collection of modelling efforts utilized in the production of analytics-based hockey content shared online.  Cloning and running the repository, given that the input play-by-play data is obtained through the wsba_hockey Python package, should mostly reproduce this content.  Some files are ommitted due to Git's file size limits or reference proprietary data.

### Models

- Expected Goals (WSBA-xG): A model for predicting the probability of a Fenwick (unblocked) shot attempt converting into a goal.  Due to recent changes in NHL tracking data, including the addition of special shot types such as "short" and "behind-net" and the automation of shot locations, the latest xG model is trained on data from the 2023-24 and 2024-25 seasons only.  

- Regularized Adjusted Plus-Minus (WSBA-RAPM): Ridge regression models developed to "isolate" contextual factors for individual contributions to team outcomes, using on-ice shot rates, penalty rates, and individual unblocked shot attempts as targets for the regressions.  Model is ran for each season.  Coefficients, including the shot map coefficients for each team, are included in the repository.

- Goals Above Replacement (WSBA-GAR): Implementation of RAPM models expanded by a player's observed time on ice for their "total" value, using an adjustment from the league-average to express the value of a player's individual contributions in terms of goals, above a replacement level NHL (which are somewhat below average skaters or goaltenders).

- Game Score (WSBA-GS): A basic, primitive Random Forest model primarily designed to describe single-game performances by skater and goaltender in NHL games.  Model is trained on regular season games from 2021-22 to 2024-25.  Model outputs are largely used to estimate a player's game-to-game consistency.  The best player performance estimates are from the RAPM and ensuing GAR models.

- Game Simulation Engine (WSBA-GSE): A simulation engine utilizing both individual performance estimates (from the RAPM models) and observed rates in conjunction with league-based constants to simulate the outcomes of NHL games.  This model is primarily utilized to predict the likeliest sets of outcomes for NHL games, including the victor, team outputs, and individual player box-score statistics.

### Data and Utilities

Some additional files assisting in the creation of online hockey analytics content are included in the repository:

- "qoc_qot.py": Module used in conjunction with the GAR model to produce estimates for each skater's quality of competition and quality of teammates (however flawed these are, we can safely interpret these considering the efforts to "isolate" skaters from one another in all aspects of GAR).

- "nhl_obj.py": Contains the code for NHL simulation engine objects, including players (forwards, defensemen, and goaltenders), coaches, and teams.  These objects are wired with the data necessary to calculate and conduct the simulations designed to predict or ideate the outcomes of NHL games.

- "wsba_network.py": Simple module for constructing edges based on estimated or observed single-game passing data.  Primarily used for the "Offensive Network" component of the single-game pages at weaksidebreakout.com
