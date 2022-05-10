# MarchMadness

The Madness of Cloud Computing

Team 15: 
Andrew Marion, Aydan Koyles, Dallas Hutchinson

Introduction:
	Every year in March, college basketball fans across the country prepare for the most exciting, heartbreaking, intense, and amazing three weeks in all of sports. The NCAA Tournament for college basketball, better known as March Madness, is the way that college basketball determines its champion each season. At the conclusion of the regular season, 64 teams are selected and placed in a bracket for the chance to compete for the title of the best college basketball team. The bracket is split into four regions of 16 teams, with each team receiving a seed from 1-16, with 1 being the highest seed. The higher seeded teams are considered better and the higher the seed, the easier (theoretically) the path is to the championship game. The tournament is single-elimination, and each year the games bring new excitement and new legacies are written. Perhaps the most defining part of March Madness is what earns the tournament its nickname: the upsets. When lower seeds get the win over a higher seed, it creates an exciting story to follow and adds to the chaos that is March Madness. 
To make the tournament even more accessible to the average fan, brackets can be filled out with the predictions of who will win with a chance to compete for prizes and bragging rights. Brackets are a fun and interactive way for fans to get involved with the tournament while also competing with friends and family. March Madness also presents ample opportunities for betting with such a large number of high-profile games condensed into a short time frame (63 games in three weeks). The exciting nature of the games and the sheer quantity have led many to try and predict these games, but the task remains elusive.
Problem Description:
	As mentioned in the introduction, filling out a bracket and betting on the games is becoming increasingly popular as March Madness continues to grow into a national phenomenon. With this increased desire to predict games comes the need for a reliable way to do so. Whether it is to impress friends and family with a near-perfect bracket or reel in the cash on several bets, any fan could benefit from a model that helps to predict the games in March Madness. 
	As of right now, over 20 million brackets are filled out each year on ESPN’s site alone (several sports websites offer bracket challenges, ESPN’s is just the most popular). In addition, with the legalization of sports betting in almost every US state, the betting market for March Madness has grown to 2.5 billion dollars a year. But despite the excitement of filling out a bracket or placing a bet, the odds of predicting these games correctly are still incredibly low. In fact, there is just a 1 in 9.2 quintillion shot of filling out a perfect bracket that predicts all 63 games correctly. 
This study will attempt to create a machine learning model that is able to predict these games with more accuracy than ever before, giving fans and bettors peace of mind each year as they make their selections. By looking at a vast amount of data (which is discussed in the next section), this model will be able to predict which teams are projected to win each game. To gauge the accuracy of the model, its final results will be compared to national averages among ESPN brackets.


Description of Dataset:
	The dataset we chose was uploaded to Kaggle as a competition at https://www.kaggle.com/c/mens-march-mania-2022/data.  This is a yearly competition to make the best March Madness prediction model.  In this dataset, there are 20 files containing information on NCAA Division 1 Men’s Basketball dating back to the 1984-85 season, the first year the 64 team March Madness format was used. These files contain boxscores (single game statistics) for every Regular Season, Conference Tournament, and March Madness game. It also contains other information such as Tournament Seeds for each team, geographical data, team and conference spelling, coaching data, and Massey ranking for each team back to the 2002-03 season.
	Although the datasets contain lots of data dating back to the 1984-85 season, we chose to utilize only data since the 2002-2003 season. We chose to limit the data for a couple of reasons. On the one hand, many of the rate statistics and other advanced metrics were not widely available or even calculated until the 2002 season began. Another reason we limited the data to this timeframe is because the game of basketball has changed a lot over the past few decades and the way in which teams win games has changed with it, so some of the older results may not be as indicative of the results today.
	We used the 4 files to obtain the data we were interested in using. First, we used the Detailed Regular Season dataset to obtain the boxscores of each game to create season statistics. Second, we used the Seeds dataset to obtain the seed for each team in a given year. Third, we used the Teams dataset to obtain the team names to make the results easy to interpret. Finally, we used Tourney Compact to obtain tournament game matchups for each game that we want to use to train and test our model with. 
	As will be discussed in the following section (Methodology), there was a lot of cleaning and transforming to be completed to prepare this data for the logistic model. We also decided to use only a specific part of the dataset, in both an attempt to optimize the efficiency of the model while also selecting the factors that we thought would be the easiest to access for the average fan. The variables we chose to include will also be discussed in the Methodology section. 
	Though our model includes many of the data points considered to be the most important in predicting college basketball games, there are many other variables that could be included in future research to refine this model and potentially increase the accuracy.
Methodology:
Variable Selection
	When selecting what variables to use, we decided to use basic regular season statistics that an average fan would be able to obtain. From sources such as Basketball Reference, fans can get basic team statistics for every team in the NCAA March Madness Tournament. Thus, we selected the following variables for each team: 





Variable
Description
WinRatio
Percentage of games won
PointsPerGame
Average number of points scored per game
PointsAllowedPerGame
Average number of points allowed per game
PointsRatio
Ratio of total points scored to total opponent points scored
OTsPerGame
Average number of overtimes per game
FGPerGame
Average number of field goals made per game
FGRatio
Ratio of total field goals made to total field goals attempted
FGAPerGame
Average number of field goals made per game
FGAllowedPerGame
Average number of field goals allowed per game
FG3PerGame
Average number of 3 point field goals made per game
FG3Ratio
Ratio of total 3 point field goals made to total 3 point field goals attempted
FG3APerGame
Average number of 3 point field goals made per game
FG3AllowedPerGame
Average number of 3 point field goals allowed per game
FTPerGame
Average number of free throws made per game
FTRatio
Ratio of total free throws made to total free throws attempted
FTAPerGame
Average number of free throws made per game
FTAllowedPerGame
Average number of free throws  allowed per game
ORPerGame
Average number of offensive rebounds per game
DRPerGame
Average number of defensive rebounds per game
TRPerGame
Average number of total rebounds per game
OppORPerGame
Average number of opponents offensive rebounds per game
OppDRPerGame
Average number of opponents defensive rebounds per game
OppTRPerGame
Average number of opponents total rebounds per game
ORRatio
Ratio of total number of offensive rebounds to opponent offensive rebounds
DRRatio
Ratio of total number of defensive rebounds to opponent defensive rebounds
TRRatio
Ratio of total number of rebounds to opponent rebounds
AstPerGame
Average number of assists per game
OppAstPerGame
Average number of opponents assists per game
StlPerGame
Average number of steals per game
OppStlPerGame
Average number of opponents steals per game
TOPerGame
Average number of turnovers per game
OppTOPerGame
Average number of opponents turnovers per game
BlkPerGame
Average number of blocks per game
OppBlkPerGame
Average number of opponents blocks per game
PFPerGame
Average number of personal fouls per game
OppPFPerGame
Average number of opponents personal fouls per game
Seed
The tournament seed given to the team

Table 1. List of independent variables
*Note: All variables are the difference between the two teams playing  in the given game
Data Cleaning

	The majority of the project was dedicated to cleaning the data and creating new variables of the basic team statistics. While the data was extremely clean, the biggest challenge was collecting the statistics from each game’s boxscore to create season-long statistics for each year. Using SparkSQL, we were able to manipulate the data into usable prediction variables. In the original Kaggle files, Team 1 was the winner every time in the tournament files. To counteract this, we put every game in the dataset in the final dataframe twice, with each team as Team 1 and Team 2.  This allowed for the model to have a balanced dataset with half of the winners being Team 1 and half being Team 2.  Next, we combined each Tournament game based off the Year, Team Identification Number.  This allowed us to get a final dataframe with both teams, each team’s seed and season statistics. To allow for less variables in the model, we took the difference between Team 1 and Team 2 statistics to condense our data.
Spark Machine Learning / Logistic Regression
	We use Spark MLLib’s Logistic Regression, a binary classification algorithm, to create a machine learning algorithm to predict the winner of each March Madness game. Using the variables we chose earlier, we used a vector assembler to create a feature vector to use in Spark Logistic Regression. Using a random split, we split all NCAA March Madness games since 2003 into 80% training set and 20% testing set.  After running test data through the model we were left with the following model evaluations:
False Positive Rate
By Label
0
0.4962962962962963
1
0.13941299790356393
False Positive Rate (Overall)
0.3187003421198181
True Positive Rate
(Recall)
By Label
0
0.860587002096436
1
0.5037037037037037
True Positive Rate (Recall) (Overall)
0.6829910479199578
Precision
By Label
0
0.6364341085271318
1
0.7816091954022989
Precision (Overall)
0.7086776351711723
F-Measure
By Label
0
0.731729055258467
1
0.6126126126126126
F-Measure (Overall)
0.6724531003873072
Accuracy (Overall)
0.6829910479199579

Table 2. Model accuracy, FPR, TPR, precision, and F-1 scores


Results:
Test Model on 2022 March Madness
	As a final test, we tested our model on the 2022 March Madness (the most recent tournament). However we did not predict our model based off of a normal bracket where you predict every game before the tournament starts. We essentially reset our predictions at the end of each round. This simulated betting before the start of the next round, betting round by round. This allows us to overcome mistakes and upsets our model did not make in the previous round. Each game was added twice, once with a team as “Team1” and again with that team as “Team2”, which ensured a fair result to determine who won. For most matchups, the model predicted one team would win both times. If not, we simply averaged the two predicted probabilities for each team and assigned the winner accordingly. Figure 2 below shows the actual and predicted label outputs for 2022 predictions.
Fig. 1. Actual vs. Predicted Heatmap
Our final predictions for 2022 left us with the following results:

Fig. 2. Accuracy by round for 2022 March Madness Tournament
As you can see in this graph, the model performed pretty well on each round of the tournament, producing an overall accuracy of 0.683 (43/63 games correctly predicted). It is also noteworthy that the model predicted these 2022 games with nearly an identical accuracy rate as the one on the testing data. This is encouraging given each future season tournament will have new teams and statistics.
Our group also wanted to see if we could predict March Madness games with better accuracy than the average ESPN bracket/fan. Table 3 below shows the national average fan bracket from 2011-2018 with an average across all years of 35.2%. As noted earlier, fans fill out the entire bracket before the tournament starts and this accuracy score weights towards correct picks made in the later rounds. Still, it is blatantly obvious how inaccurate the average fan bracket is and our model may provide an avenue for increased reliability when making predictions.
2018
29.7%
2017
34.2%
2016
35.5%
2015
43.4%
2014
31.3%
2013
36.5%
2012
43.2%
2011
27.7%
Total
35.2%

Table 3. Average ESPN fan bracket accuracy by year
	Given its current deployment setup, our model may also be suitable for round-by-round sports betting on which team will win each matchup. Nationally, sports betting is a trending industry with more and more fans searching for a competitive edge when making their sports picks. Future improvements to this project would allow a user to simulate the entire tournament in the more traditional way, where all 63 tournament games are picked and locked in before any of the games start.

References
Dataset from Kaggle (https://www.kaggle.com/c/mens-march-mania-2022/data).
“March Madness: Was your 2018 bracket average?” NCAA,  https://www.ncaa.com/news/basketball-men/2019-02-27/march-madness-how-do-your-past-brackets-stack-competition
Appendix I
Andrew Marion - 
Lead Engineer for Data Collection and Data Cleaning
Report Writing
Presentation Creation
Interpret Results
Aydan Koyles - 
Lead Engineer for Transforming Data
Report Writing
Presentation Creation
Interpret Results
Dallas Hutchinson - 
Lead Engineer for Logistic Regression Model
Report Writing
Presentation Creation
Interpret Results

