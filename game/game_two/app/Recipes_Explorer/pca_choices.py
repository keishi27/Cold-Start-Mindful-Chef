import pandas as pd
import numpy as np
import random
from loguru import logger
from Recipes_Explorer.random_choices import ChooseRandomPCAWeightedProb



def choose_pca(recipesData, numChoices, foodGroupChoice):
    vegan_df = pd.read_csv("../../data/Vegan_FG_Scaled_PC.csv")
    chicken_df = pd.read_csv("../../data/Chicken_FG_Scaled_PC.csv")
    fish_df = pd.read_csv("../../data/Fish_FG_Scaled_PC.csv")
    beef_df = pd.read_csv("../../data/Beef_FG_Scaled_PC.csv")
    pork_df = pd.read_csv("../../data/Pork_FG_Scaled_PC.csv")
    lamb_df = pd.read_csv("../../data/Lamb_FG_Scaled_PC.csv")

    arrayRecipeID = np.array([])

    if foodGroupChoice == [0, 0, 0, 0, 0, "Vegan"]:
        vegan_pc1_33 = np.array(
            vegan_df[vegan_df["FG_Scaled_PC_1"] < 0.33].sample()["id"]
        )
        vegan_pc1_66 = np.array(
            vegan_df[
                (vegan_df["FG_Scaled_PC_1"] > 0.33)
                & (vegan_df["FG_Scaled_PC_1"] < 0.66)
            ].sample(2)["id"]
        )
        vegan_pc1_66_100 = np.array(
            vegan_df[vegan_df["FG_Scaled_PC_1"] > 0.66].sample()["id"]
        )
        vegan_pc2_33 = np.array(
            vegan_df[vegan_df["FG_Scaled_PC_2"] < 0.33].sample()["id"]
        )
        vegan_pc2_66 = np.array(
            vegan_df[
                (vegan_df["FG_Scaled_PC_2"] > 0.33)
                & (vegan_df["FG_Scaled_PC_2"] < 0.66)
            ].sample()["id"]
        )
        vegan_pc2_66_100 = np.array(
            vegan_df[vegan_df["FG_Scaled_PC_2"] > 0.66].sample()["id"]
        )
        vegan_pc3_33 = np.array(
            vegan_df[vegan_df["FG_Scaled_PC_3"] < 0.33].sample()["id"]
        )
        vegan_pc3_66 = np.array(
            vegan_df[
                (vegan_df["FG_Scaled_PC_3"] > 0.33)
                & (vegan_df["FG_Scaled_PC_3"] < 0.66)
            ].sample()["id"]
        )
        vegan_pc3_66_100 = np.array(
            vegan_df[vegan_df["FG_Scaled_PC_3"] > 0.66].sample()["id"]
        )

        arrayRecipeID = np.append(
            arrayRecipeID,
            np.concatenate(
                (
                    vegan_pc1_33,
                    vegan_pc1_66,
                    vegan_pc1_66_100,
                    vegan_pc2_33,
                    vegan_pc2_66,
                    vegan_pc2_66_100,
                    vegan_pc3_33,
                    vegan_pc3_66,
                    vegan_pc3_66_100,
                )
            ),
        )

    else:
        vegan_pc1_50 = vegan_df[vegan_df["FG_Scaled_PC_1"] < 0.5]
        vegan_pc1_100 = vegan_df[vegan_df["FG_Scaled_PC_1"] > 0.5]
        chicken_pc1_50 = chicken_df[chicken_df["FG_Scaled_PC_1"] < 0.5]
        chicken_pc1_100 = chicken_df[chicken_df["FG_Scaled_PC_1"] > 0.5]
        fish_pc1_50 = fish_df[fish_df["FG_Scaled_PC_1"] < 0.5]
        fish_pc1_100 = fish_df[fish_df["FG_Scaled_PC_1"] > 0.5]
        beef_pc1_50 = beef_df[beef_df["FG_Scaled_PC_1"] < 0.5]
        beef_pc1_100 = beef_df[beef_df["FG_Scaled_PC_1"] > 0.5]
        pork_pc1_50 = pork_df[pork_df["FG_Scaled_PC_1"] < 0.5]
        pork_pc1_100 = pork_df[pork_df["FG_Scaled_PC_1"] > 0.5]
        lamb_pc1_50 = lamb_df[lamb_df["FG_Scaled_PC_1"] < 0.5]
        lamb_pc1_100 = lamb_df[lamb_df["FG_Scaled_PC_1"] > 0.5]

        vegan_pc1_50["weight"], vegan_pc1_100["weight"] = 1, 1
        chicken_pc1_50["weight"], chicken_pc1_100["weight"] = 1, 1
        fish_pc1_50["weight"], fish_pc1_100["weight"] = 1, 1
        beef_pc1_50["weight"], beef_pc1_100["weight"] = 1, 1
        pork_pc1_50["weight"], pork_pc1_100["weight"] = 1, 1
        lamb_pc1_50["weight"], lamb_pc1_100["weight"] = 1, 1

        arrayRecipeID = np.array([])
        count = 0
        while count < numChoices:
            for elem in foodGroupChoice:
                if elem == "Chicken":
                    if count == numChoices:
                        break
                    chicken_pc1_50_samp = ChooseRandomPCAWeightedProb(
                        np.array(chicken_pc1_50["id"]),
                        np.array(chicken_pc1_50["weight"]),
                        1,
                    )
                    chicken_pc1_100_samp = ChooseRandomPCAWeightedProb(
                        np.array(chicken_pc1_100["id"]),
                        np.array(chicken_pc1_100["weight"]),
                        1,
                    )
                    chicken_pc1_50.loc[chicken_pc1_50["id"] == chicken_pc1_50_samp[0]][
                        "weight"
                    ] = 0
                    chicken_pc1_100.loc[
                        chicken_pc1_100["id"] == chicken_pc1_100_samp[0]
                    ]["weight"] = 0
                    arrayRecipeID = np.append(
                        arrayRecipeID,
                        np.concatenate((chicken_pc1_50_samp, chicken_pc1_100_samp)),
                    )
                    count += 2

                if elem == "Fish":
                    if count == numChoices:
                        break
                    fish_pc1_50_samp = ChooseRandomPCAWeightedProb(
                        np.array(fish_pc1_50["id"]), np.array(fish_pc1_50["weight"]), 1
                    )
                    fish_pc1_100_samp = ChooseRandomPCAWeightedProb(
                        np.array(fish_pc1_100["id"]),
                        np.array(fish_pc1_100["weight"]),
                        1,
                    )
                    fish_pc1_50.loc[fish_pc1_50["id"] == fish_pc1_50_samp[0]][
                        "weight"
                    ] = 0
                    fish_pc1_100.loc[fish_pc1_100["id"] == fish_pc1_100_samp[0]][
                        "weight"
                    ] = 0
                    arrayRecipeID = np.append(
                        arrayRecipeID,
                        np.concatenate((fish_pc1_50_samp, fish_pc1_100_samp)),
                    )
                    count += 2

                if elem == "Beef":
                    if count == numChoices:
                        break
                    beef_pc1_50_samp = ChooseRandomPCAWeightedProb(
                        np.array(beef_pc1_50["id"]), np.array(beef_pc1_50["weight"]), 1
                    )
                    beef_pc1_100_samp = ChooseRandomPCAWeightedProb(
                        np.array(beef_pc1_100["id"]),
                        np.array(beef_pc1_100["weight"]),
                        1,
                    )
                    beef_pc1_50.loc[beef_pc1_50["id"] == beef_pc1_50_samp[0]][
                        "weight"
                    ] = 0
                    beef_pc1_100.loc[beef_pc1_100["id"] == beef_pc1_100_samp[0]][
                        "weight"
                    ] = 0
                    arrayRecipeID = np.append(
                        arrayRecipeID,
                        np.concatenate((beef_pc1_50_samp, beef_pc1_100_samp)),
                    )
                    count += 2

                if elem == "Pork":
                    if count == numChoices:
                        break
                    pork_pc1_50_samp = ChooseRandomPCAWeightedProb(
                        np.array(pork_pc1_50["id"]), np.array(pork_pc1_50["weight"]), 1
                    )
                    pork_pc1_100_samp = ChooseRandomPCAWeightedProb(
                        np.array(pork_pc1_100["id"]),
                        np.array(pork_pc1_100["weight"]),
                        1,
                    )
                    pork_pc1_50.loc[pork_pc1_50["id"] == pork_pc1_50_samp[0]][
                        "weight"
                    ] = 0
                    pork_pc1_100.loc[pork_pc1_100["id"] == pork_pc1_100_samp[0]][
                        "weight"
                    ] = 0
                    arrayRecipeID = np.append(
                        arrayRecipeID,
                        np.concatenate((pork_pc1_50_samp, pork_pc1_100_samp)),
                    )
                    count += 2

                if elem == "Lamb":
                    if count == numChoices:
                        break
                    lamb_pc1_50_samp = ChooseRandomPCAWeightedProb(
                        np.array(lamb_pc1_50["id"]), np.array(lamb_pc1_50["weight"]), 1
                    )
                    lamb_pc1_100_samp = ChooseRandomPCAWeightedProb(
                        np.array(lamb_pc1_100["id"]),
                        np.array(lamb_pc1_100["weight"]),
                        1,
                    )
                    lamb_pc1_50.loc[lamb_pc1_50["id"] == lamb_pc1_50_samp[0]][
                        "weight"
                    ] = 0
                    lamb_pc1_100.loc[lamb_pc1_100["id"] == lamb_pc1_100_samp[0]][
                        "weight"
                    ] = 0
                    arrayRecipeID = np.append(
                        arrayRecipeID,
                        np.concatenate((lamb_pc1_50_samp, lamb_pc1_100_samp)),
                    )
                    count += 2

                if elem == "Vegan":
                    if count == numChoices:
                        break
                    vegan_pc1_50_samp = ChooseRandomPCAWeightedProb(
                        np.array(vegan_pc1_50["id"]),
                        np.array(vegan_pc1_50["weight"]),
                        1,
                    )
                    vegan_pc1_100_samp = ChooseRandomPCAWeightedProb(
                        np.array(vegan_pc1_100["id"]),
                        np.array(vegan_pc1_100["weight"]),
                        1,
                    )
                    vegan_pc1_50.loc[vegan_pc1_50["id"] == vegan_pc1_50_samp[0]][
                        "weight"
                    ] = 0
                    vegan_pc1_100.loc[vegan_pc1_100["id"] == vegan_pc1_100_samp[0]][
                        "weight"
                    ] = 0
                    arrayRecipeID = np.append(
                        arrayRecipeID,
                        np.concatenate((vegan_pc1_50_samp, vegan_pc1_100_samp)),
                    )
                    count += 2

    return arrayRecipeID


