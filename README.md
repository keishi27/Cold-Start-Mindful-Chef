## Description
- Repo for team project, developing cold-start problem solution for food-box start-up
- Working with Mindful Chef and Science to Data Science (MC/S2DS).
- MC required a way to profile new customers and their tastes, to best recommend recipes.
- Our team developed 3 onboarding tools, driven by ML techniques.
- Game 1 acted as the baseline model (MC current system).
- Game 2 used NLP and dimensionality reduction to select recipes for onboarding tool.
- Game 3 used topic modelling and clustering to select recipes.
- Onboarding tools were sent to 21,000 customers for A/B testing, with a 11%+ response rate.
- Using statistical analysis, both Games 2 and 3 were found to be better than Game 1 at predicting customer tastes.

## Table of Contents

**data**: Store all csv files here. Analysis outputs are also generated here.

**flaskapp_andrius**: Wrapper and code to generate delta12.

**game_one**: Front and back-end code for Game 1 (baseline).

**game_three**: Front and back-end code for Game 3 (clustering).

**game_two**: Front and back-end code for Game 2 (extremes).

**notebooks**:
1. Exploratory data analysis.
2. Game 2 development: Dimensionality reduction of recipe table and generation of CSVs for Game 2. EDA into single customer journey in foodspace
3. Generate clean recipe table used in all games, fix image errors, set up analysis metrics and test calculations using sample customer database.
4. Analysis of data from A/B testing, comparison with historical customer data. Statistical analysis of results.
5. Clustering recipes using topic modelling and K-means, generating the csv file for Game 3.
6. Aditional attempts to cluster recipes.

**reports**: Updates and deliverables presented to S2DS and Mindful Chef.

**src**: Source code for project.

## Installation 
1. Create local environment: 
- `conda env create -f environment.yml`