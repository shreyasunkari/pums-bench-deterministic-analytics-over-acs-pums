from pathlib import Path
import pandas as pd
from load_data import load_person, load_household
from utils import (
    round_half_away_from_zero,
    weighted_quantile,
    sdr_standard_error,
)


def solve_q001_q006():
    # Load only the columns needed for q001-q006.
    # Reading a subset of columns still preserves the number of rows.
    people = load_person(
        2024,
        usecols=["AGEP", "SEX", "ESR"],
    )

    households = load_household(
        2024,
        usecols=["TYPEHUGQ", "NP"],
    )

    answers = {}

    # q001: total number of person records
    answers["q001"] = len(people)

    # q002: people age 65 or older
    answers["q002"] = (people["AGEP"] >= 65).sum()

    # q003: household-file records that are housing units
    answers["q003"] = (households["TYPEHUGQ"] == 1).sum()

    # q004: female person records
    answers["q004"] = (people["SEX"] == 2).sum()

    # q005: unemployed person records
    # ESR blanks become NaN, so they naturally do not equal 3.
    answers["q005"] = (people["ESR"] == 3).sum()

    # q006: vacant housing units
    answers["q006"] = (
        (households["TYPEHUGQ"] == 1)
        & (households["NP"] == 0)
    ).sum()

    return answers


def solve_q007_q010():
    people = load_person(
        2024,
        usecols=[
            "AGEP",
            "PWGTP",
            "ESR",
            "WAGP",
            "ADJINC",
        ],
    )

    households = load_household(
        2024,
        usecols=[
            "TYPEHUGQ",
            "NP",
            "TEN",
            "WGTP",
        ],
    )

    answers = {}

    # q007:
    # Estimated number of people age 65+
    age_65_plus = people["AGEP"] >= 65

    answers["q007"] = int(
        people.loc[age_65_plus, "PWGTP"].sum()
    )

    # q008:
    # Estimated number of renter-occupied households
    renters = (
        (households["TYPEHUGQ"] == 1)
        & (households["NP"] >= 1)
        & (households["TEN"] == 3)
    )

    answers["q008"] = int(
        households.loc[renters, "WGTP"].sum()
    )

    # q009:
    # PWGTP-weighted mean age
    weighted_age_sum = (
        people["AGEP"] * people["PWGTP"]
    ).sum()

    total_person_weight = people["PWGTP"].sum()

    mean_age = weighted_age_sum / total_person_weight

    answers["q009"] = str(
        round_half_away_from_zero(mean_age, 2)
    )

    # q010:
    # Civilian employed persons
    employed = people["ESR"].isin([1, 2])

    employed_people = people.loc[employed].copy()

    # Convert reported wages into ADJINC-adjusted dollars
    employed_people["adjusted_wage"] = (
        employed_people["WAGP"]
        * employed_people["ADJINC"]
        / 1_000_000
    )

    weighted_wage_sum = (
        employed_people["adjusted_wage"]
        * employed_people["PWGTP"]
    ).sum()

    employed_weight = employed_people["PWGTP"].sum()

    mean_wage = weighted_wage_sum / employed_weight

    answers["q010"] = str(
        round_half_away_from_zero(mean_wage, 0)
    )

    return answers


def solve_q011_q017():
    people = load_person(
        2024,
        usecols=[
            "AGEP",
            "PWGTP",
            "ESR",
            "WAGP",
            "ADJINC",
            "SCHL",
        ],
    )

    households = load_household(
        2024,
        usecols=[
            "TYPEHUGQ",
            "NP",
            "WGTP",
            "HINCP",
            "ADJINC",
        ],
    )

    answers = {}

    # q011:
    # PWGTP-weighted median age over all people
    median_age = weighted_quantile(
        people["AGEP"],
        people["PWGTP"],
        0.50,
    )

    answers["q011"] = int(median_age)

    # q012:
    # Weighted median wage among civilian employed persons
    # with positive wage income.
    employed_positive_wage = (
        people["ESR"].isin([1, 2])
        & (people["WAGP"] > 0)
    )

    wage_population = people.loc[employed_positive_wage]

    median_wage = weighted_quantile(
        wage_population["WAGP"],
        wage_population["PWGTP"],
        0.50,
    )

    # ADJINC should be constant within this ACS release.
    assert people["ADJINC"].nunique() == 1
    person_adjinc = people["ADJINC"].iloc[0] / 1_000_000

    adjusted_median_wage = median_wage * person_adjinc

    answers["q012"] = str(
        round_half_away_from_zero(adjusted_median_wage, 0)
    )

    # q013:
    # WGTP-weighted median household income among occupied units
    occupied = (
        (households["TYPEHUGQ"] == 1)
        & (households["NP"] >= 1)
    )

    occupied_households = households.loc[occupied]

    median_household_income = weighted_quantile(
        occupied_households["HINCP"],
        occupied_households["WGTP"],
        0.50,
    )

    assert households["ADJINC"].nunique() == 1
    household_adjinc = households["ADJINC"].iloc[0] / 1_000_000

    adjusted_median_household_income = (
        median_household_income * household_adjinc
    )

    answers["q013"] = str(
        round_half_away_from_zero(
            adjusted_median_household_income,
            0,
        )
    )

    # q014:
    # PWGTP-weighted 75th percentile of age
    age_75th = weighted_quantile(
        people["AGEP"],
        people["PWGTP"],
        0.75,
    )

    answers["q014"] = int(age_75th)

    # q015:
    # WGTP-weighted mean household size among occupied units
    weighted_household_size = (
        occupied_households["NP"]
        * occupied_households["WGTP"]
    ).sum()

    total_household_weight = occupied_households["WGTP"].sum()

    mean_household_size = (
        weighted_household_size / total_household_weight
    )

    answers["q015"] = str(
        round_half_away_from_zero(mean_household_size, 2)
    )

    # q016:
    # Weighted percentage of Texas population under age 18
    under_18_weight = people.loc[
        people["AGEP"] < 18,
        "PWGTP",
    ].sum()

    total_population_weight = people["PWGTP"].sum()

    under_18_percent = (
        100 * under_18_weight / total_population_weight
    )

    answers["q016"] = str(
        round_half_away_from_zero(under_18_percent, 1)
    )

    # q017:
    # Among people age 25+, percentage with bachelor's degree+
    age_25_plus = people["AGEP"] >= 25

    bachelors_plus = (
        age_25_plus
        & (people["SCHL"] >= 21)
    )

    bachelors_plus_weight = people.loc[
        bachelors_plus,
        "PWGTP",
    ].sum()

    age_25_plus_weight = people.loc[
        age_25_plus,
        "PWGTP",
    ].sum()

    bachelors_plus_percent = (
        100 * bachelors_plus_weight / age_25_plus_weight
    )

    answers["q017"] = str(
        round_half_away_from_zero(bachelors_plus_percent, 1)
    )

    return answers


def solve_q018_q026():
    people = load_person(
        2024,
        usecols=[
            "SERIALNO",
            "SPORDER",
            "AGEP",
            "SEX",
            "PWGTP",
        ],
    )

    households = load_household(
        2024,
        usecols=[
            "SERIALNO",
            "TYPEHUGQ",
            "NP",
            "TEN",
            "WGTP",
            "HINCP",
            "ADJINC",
        ],
    )

    answers = {}

    # ---------------------------------------------------------
    # Person -> household join
    # Each household record may join to many people,
    # but each person should join to exactly one household record.
    # ---------------------------------------------------------
    person_household = people.merge(
        households[
            [
                "SERIALNO",
                "TYPEHUGQ",
                "NP",
                "TEN",
                "HINCP",
                "ADJINC",
            ]
        ],
        on="SERIALNO",
        how="left",
        validate="many_to_one",
    )

    # q018:
    # Estimated persons living in renter-occupied housing units
    renter_persons = (
        (person_household["TYPEHUGQ"] == 1)
        & (person_household["NP"] >= 1)
        & (person_household["TEN"] == 3)
    )

    answers["q018"] = int(
        person_household.loc[
            renter_persons,
            "PWGTP",
        ].sum()
    )

    # q019:
    # Person-weighted median household income experienced by persons
    occupied_persons = (
        (person_household["TYPEHUGQ"] == 1)
        & (person_household["NP"] >= 1)
    )

    occupied_person_df = person_household.loc[
        occupied_persons
    ]

    person_weighted_median_hincp = weighted_quantile(
        occupied_person_df["HINCP"],
        occupied_person_df["PWGTP"],
        0.50,
    )

    assert occupied_person_df["ADJINC"].nunique() == 1

    adjinc = (
        occupied_person_df["ADJINC"].iloc[0]
        / 1_000_000
    )

    answers["q019"] = str(
        round_half_away_from_zero(
            person_weighted_median_hincp * adjinc,
            0,
        )
    )

    # q020:
    # Person-weighted mean household size experienced by a person
    mean_household_size_person_weighted = (
        (
            occupied_person_df["NP"]
            * occupied_person_df["PWGTP"]
        ).sum()
        / occupied_person_df["PWGTP"].sum()
    )

    answers["q020"] = str(
        round_half_away_from_zero(
            mean_household_size_person_weighted,
            2,
        )
    )

    # ---------------------------------------------------------
    # Householder table
    # SPORDER == 1 identifies the householder.
    # Keep one person record per SERIALNO.
    # ---------------------------------------------------------
    householders = people.loc[
        people["SPORDER"] == 1,
        ["SERIALNO", "AGEP", "SEX"],
    ].rename(
        columns={
            "AGEP": "HOUSEHOLDER_AGE",
            "SEX": "HOUSEHOLDER_SEX",
        }
    )

    household_with_householder = households.merge(
        householders,
        on="SERIALNO",
        how="left",
        validate="one_to_one",
    )

    occupied_households = (
        (household_with_householder["TYPEHUGQ"] == 1)
        & (household_with_householder["NP"] >= 1)
    )

    occupied_hh = household_with_householder.loc[
        occupied_households
    ]

    # q021:
    # Estimated occupied housing units with female householder
    female_householder = (
        occupied_hh["HOUSEHOLDER_SEX"] == 2
    )

    answers["q021"] = int(
        occupied_hh.loc[
            female_householder,
            "WGTP",
        ].sum()
    )

    # q022:
    # Household-weighted median householder age
    median_householder_age = weighted_quantile(
        occupied_hh["HOUSEHOLDER_AGE"],
        occupied_hh["WGTP"],
        0.50,
    )

    answers["q022"] = int(median_householder_age)

    # q023:
    # Estimated occupied housing units with >=1 member age 65+
    has_65_plus = (
        people.assign(
            IS_65_PLUS=people["AGEP"] >= 65
        )
        .groupby("SERIALNO")["IS_65_PLUS"]
        .any()
        .rename("HAS_65_PLUS")
        .reset_index()
    )

    household_age_flag = households.merge(
        has_65_plus,
        on="SERIALNO",
        how="left",
        validate="one_to_one",
    )

    qualifying_65_households = (
        (household_age_flag["TYPEHUGQ"] == 1)
        & (household_age_flag["NP"] >= 1)
        & (household_age_flag["HAS_65_PLUS"] == True)
    )

    answers["q023"] = int(
        household_age_flag.loc[
            qualifying_65_households,
            "WGTP",
        ].sum()
    )

    # q024:
    # Estimated persons living in group quarters
    group_quarters_persons = (
        person_household["TYPEHUGQ"].isin([2, 3])
    )

    answers["q024"] = int(
        person_household.loc[
            group_quarters_persons,
            "PWGTP",
        ].sum()
    )

    # q025:
    # Percentage of occupied-unit persons living in owner-occupied units
    owner_persons = (
        occupied_persons
        & person_household["TEN"].isin([1, 2])
    )

    owner_weight = person_household.loc[
        owner_persons,
        "PWGTP",
    ].sum()

    occupied_person_weight = person_household.loc[
        occupied_persons,
        "PWGTP",
    ].sum()

    owner_percent = (
        100 * owner_weight / occupied_person_weight
    )

    answers["q025"] = str(
        round_half_away_from_zero(
            owner_percent,
            1,
        )
    )

    # q026:
    # Household-weighted median household income
    # for occupied units whose householder is 65+
    senior_householder_hh = occupied_hh.loc[
        occupied_hh["HOUSEHOLDER_AGE"] >= 65
    ]

    senior_median_hincp = weighted_quantile(
        senior_householder_hh["HINCP"],
        senior_householder_hh["WGTP"],
        0.50,
    )

    assert senior_householder_hh["ADJINC"].nunique() == 1

    senior_adjinc = (
        senior_householder_hh["ADJINC"].iloc[0]
        / 1_000_000
    )

    answers["q026"] = str(
        round_half_away_from_zero(
            senior_median_hincp * senior_adjinc,
            0,
        )
    )

    return answers


def solve_q027_q033():
    # ----------------------------
    # 2024 household data
    # ----------------------------
    households_2024 = load_household(
        2024,
        usecols=[
            "TYPEHUGQ",
            "NP",
            "TEN",
            "WGTP",
            "GRNTP",
            "VALP",
            "ADJHSG",
            "HINCP",
            "ADJINC",
        ],
    )

    # ----------------------------
    # 2023 data
    # ----------------------------
    people_2023 = load_person(
        2023,
        usecols=[
            "AGEP",
            "PWGTP",
        ],
    )

    households_2023 = load_household(
        2023,
        usecols=[
            "TYPEHUGQ",
            "NP",
            "TEN",
            "WGTP",
            "HINCP",
            "ADJINC",
        ],
    )

    # Need 2024 population for q030
    people_2024 = load_person(
        2024,
        usecols=["PWGTP"],
    )

    answers = {}

    # =========================================================
    # q027
    # Weighted median gross rent, adjusted using ADJHSG
    # =========================================================
    gross_rent_universe = (
        (households_2024["TYPEHUGQ"] == 1)
        & (households_2024["NP"] >= 1)
        & households_2024["GRNTP"].notna()
    )

    gross_rent_df = households_2024.loc[gross_rent_universe]

    median_gross_rent = weighted_quantile(
        gross_rent_df["GRNTP"],
        gross_rent_df["WGTP"],
        0.50,
    )

    assert households_2024["ADJHSG"].nunique() == 1

    adjhsg_2024 = (
        households_2024["ADJHSG"].iloc[0]
        / 1_000_000
    )

    adjusted_median_gross_rent = (
        median_gross_rent * adjhsg_2024
    )

    answers["q027"] = str(
        round_half_away_from_zero(
            adjusted_median_gross_rent,
            0,
        )
    )

    # =========================================================
    # q028
    # Weighted median property value among owner-occupied units
    # =========================================================
    owner_property_universe = (
        (households_2024["TYPEHUGQ"] == 1)
        & (households_2024["NP"] >= 1)
        & households_2024["TEN"].isin([1, 2])
        & households_2024["VALP"].notna()
    )

    owner_property_df = households_2024.loc[
        owner_property_universe
    ]

    median_property_value = weighted_quantile(
        owner_property_df["VALP"],
        owner_property_df["WGTP"],
        0.50,
    )

    adjusted_median_property_value = (
        median_property_value * adjhsg_2024
    )

    answers["q028"] = str(
        round_half_away_from_zero(
            adjusted_median_property_value,
            0,
        )
    )

    # =========================================================
    # q029
    # 2023 weighted median age
    # =========================================================
    median_age_2023 = weighted_quantile(
        people_2023["AGEP"],
        people_2023["PWGTP"],
        0.50,
    )

    answers["q029"] = int(median_age_2023)

    # =========================================================
    # q030
    # Texas population growth: 2024 - 2023
    # =========================================================
    population_2024 = people_2024["PWGTP"].sum()
    population_2023 = people_2023["PWGTP"].sum()

    answers["q030"] = int(
        population_2024 - population_2023
    )

    # =========================================================
    # q031
    # 2023 weighted median household income
    # =========================================================
    occupied_2023 = (
        (households_2023["TYPEHUGQ"] == 1)
        & (households_2023["NP"] >= 1)
    )

    occupied_hh_2023 = households_2023.loc[
        occupied_2023
    ]

    median_hincp_2023 = weighted_quantile(
        occupied_hh_2023["HINCP"],
        occupied_hh_2023["WGTP"],
        0.50,
    )

    assert households_2023["ADJINC"].nunique() == 1

    adjinc_2023 = (
        households_2023["ADJINC"].iloc[0]
        / 1_000_000
    )

    adjusted_median_hincp_2023 = (
        median_hincp_2023 * adjinc_2023
    )

    answers["q031"] = str(
        round_half_away_from_zero(
            adjusted_median_hincp_2023,
            0,
        )
    )

    # =========================================================
    # q032
    # Percent change in adjusted median household income
    # =========================================================
    occupied_2024 = (
        (households_2024["TYPEHUGQ"] == 1)
        & (households_2024["NP"] >= 1)
    )

    occupied_hh_2024 = households_2024.loc[
        occupied_2024
    ]

    median_hincp_2024 = weighted_quantile(
        occupied_hh_2024["HINCP"],
        occupied_hh_2024["WGTP"],
        0.50,
    )

    assert households_2024["ADJINC"].nunique() == 1

    adjinc_2024 = (
        households_2024["ADJINC"].iloc[0]
        / 1_000_000
    )

    adjusted_median_hincp_2024 = (
        median_hincp_2024 * adjinc_2024
    )

    income_percent_change = (
        100
        * (
            adjusted_median_hincp_2024
            - adjusted_median_hincp_2023
        )
        / adjusted_median_hincp_2023
    )

    answers["q032"] = str(
        round_half_away_from_zero(
            income_percent_change,
            1,
        )
    )

    # =========================================================
    # q033
    # Change in renter-occupied households: 2024 - 2023
    # =========================================================
    renters_2023 = (
        (households_2023["TYPEHUGQ"] == 1)
        & (households_2023["NP"] >= 1)
        & (households_2023["TEN"] == 3)
    )

    renters_2024 = (
        (households_2024["TYPEHUGQ"] == 1)
        & (households_2024["NP"] >= 1)
        & (households_2024["TEN"] == 3)
    )

    renter_total_2023 = households_2023.loc[
        renters_2023,
        "WGTP",
    ].sum()

    renter_total_2024 = households_2024.loc[
        renters_2024,
        "WGTP",
    ].sum()

    answers["q033"] = int(
        renter_total_2024 - renter_total_2023
    )

    return answers


def solve_q034_q038():
    person_replicate_weights = [
        f"PWGTP{i}" for i in range(1, 81)
    ]

    household_replicate_weights = [
        f"WGTP{i}" for i in range(1, 81)
    ]

    people = load_person(
        2024,
        usecols=[
            "AGEP",
            "SCHL",
            "ESR",
            "WAGP",
            "ADJINC",
            "PWGTP",
            *person_replicate_weights,
        ],
    )

    households = load_household(
        2024,
        usecols=[
            "TYPEHUGQ",
            "NP",
            "TEN",
            "HINCP",
            "ADJINC",
            "WGTP",
            *household_replicate_weights,
        ],
    )

    answers = {}

    # =========================================================
    # q034
    # SE of estimated number of persons age 65+
    # =========================================================
    age_65_plus = people["AGEP"] >= 65

    age_65_df = people.loc[age_65_plus]

    full_age_65_estimate = age_65_df["PWGTP"].sum()

    replicate_age_65_estimates = [
        age_65_df[weight].sum()
        for weight in person_replicate_weights
    ]

    age_65_se = sdr_standard_error(
        full_age_65_estimate,
        replicate_age_65_estimates,
    )

    answers["q034"] = str(
        round_half_away_from_zero(age_65_se, 0)
    )

    # =========================================================
    # q035
    # SE of estimated number of renter households
    # =========================================================
    renter_universe = (
        (households["TYPEHUGQ"] == 1)
        & (households["NP"] >= 1)
        & (households["TEN"] == 3)
    )

    renter_df = households.loc[renter_universe]

    full_renter_estimate = renter_df["WGTP"].sum()

    replicate_renter_estimates = [
        renter_df[weight].sum()
        for weight in household_replicate_weights
    ]

    renter_se = sdr_standard_error(
        full_renter_estimate,
        replicate_renter_estimates,
    )

    answers["q035"] = str(
        round_half_away_from_zero(renter_se, 0)
    )

    # =========================================================
    # q036
    # SE of weighted mean adjusted wage
    # =========================================================
    employed = people["ESR"].isin([1, 2])

    employed_df = people.loc[employed].copy()

    employed_df["ADJUSTED_WAGE"] = (
        employed_df["WAGP"]
        * employed_df["ADJINC"]
        / 1_000_000
    )

    full_mean_wage = (
        (
            employed_df["ADJUSTED_WAGE"]
            * employed_df["PWGTP"]
        ).sum()
        / employed_df["PWGTP"].sum()
    )

    replicate_mean_wages = []

    for weight in person_replicate_weights:
        replicate_mean = (
            (
                employed_df["ADJUSTED_WAGE"]
                * employed_df[weight]
            ).sum()
            / employed_df[weight].sum()
        )

        replicate_mean_wages.append(replicate_mean)

    mean_wage_se = sdr_standard_error(
        full_mean_wage,
        replicate_mean_wages,
    )

    answers["q036"] = str(
        round_half_away_from_zero(mean_wage_se, 0)
    )

    # =========================================================
    # q037
    # SE of weighted median adjusted household income
    # =========================================================
    occupied = (
        (households["TYPEHUGQ"] == 1)
        & (households["NP"] >= 1)
    )

    occupied_df = households.loc[occupied].copy()

    # HINCP should be defined for this universe.
    assert occupied_df["HINCP"].notna().all()

    assert occupied_df["ADJINC"].nunique() == 1

    adjinc = (
        occupied_df["ADJINC"].iloc[0]
        / 1_000_000
    )

    full_median_income = weighted_quantile(
        occupied_df["HINCP"],
        occupied_df["WGTP"],
        0.50,
    )

    full_adjusted_median_income = (
        full_median_income * adjinc
    )

    replicate_median_incomes = []

    for weight in household_replicate_weights:
        replicate_median = weighted_quantile(
            occupied_df["HINCP"],
            occupied_df[weight],
            0.50,
        )

        replicate_median_incomes.append(
            replicate_median * adjinc
        )

    median_income_se = sdr_standard_error(
        full_adjusted_median_income,
        replicate_median_incomes,
    )

    answers["q037"] = str(
        round_half_away_from_zero(
            median_income_se,
            0,
        )
    )

    # =========================================================
    # q038
    # SE of percentage age 25+ with bachelor's degree+
    # =========================================================
    age_25_plus = people["AGEP"] >= 25

    bachelors_plus = (
        age_25_plus
        & (people["SCHL"] >= 21)
    )

    full_25_plus_weight = people.loc[
        age_25_plus,
        "PWGTP",
    ].sum()

    full_bachelors_weight = people.loc[
        bachelors_plus,
        "PWGTP",
    ].sum()

    full_bachelors_percent = (
        100
        * full_bachelors_weight
        / full_25_plus_weight
    )

    replicate_bachelors_percents = []

    for weight in person_replicate_weights:
        replicate_denominator = people.loc[
            age_25_plus,
            weight,
        ].sum()

        replicate_numerator = people.loc[
            bachelors_plus,
            weight,
        ].sum()

        replicate_percent = (
            100
            * replicate_numerator
            / replicate_denominator
        )

        replicate_bachelors_percents.append(
            replicate_percent
        )

    bachelors_percent_se = sdr_standard_error(
        full_bachelors_percent,
        replicate_bachelors_percents,
    )

    answers["q038"] = str(
        round_half_away_from_zero(
            bachelors_percent_se,
            2,
        )
    )

    return answers


def write_submission(answers):
    project_root = Path(__file__).resolve().parents[1]

    sample_path = (
        project_root
        / "data"
        / "competition"
        / "sample_submission.csv"
    )

    output_path = (
        project_root
        / "outputs"
        / "submission.csv"
    )

    submission = pd.read_csv(
        sample_path,
        dtype={"question_id": "string", "answer": "string"},
    )

    submission["answer"] = submission["question_id"].map(
        lambda qid: str(answers[qid])
    )

    if submission["answer"].isna().any():
        missing = submission.loc[
            submission["answer"].isna(),
            "question_id",
        ].tolist()

        raise ValueError(
            f"Missing answers for: {missing}"
        )

    submission.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Submission written to: {output_path}")


if __name__ == "__main__":
    answers = {}

    answers.update(solve_q001_q006())
    answers.update(solve_q007_q010())
    answers.update(solve_q011_q017())
    answers.update(solve_q018_q026())
    answers.update(solve_q027_q033())
    answers.update(solve_q034_q038())

    for question_id, answer in answers.items():
        print(f"{question_id}: {answer}")

    assert len(answers) == 38

    write_submission(answers)
