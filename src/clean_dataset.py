import pandas as pd

non_postpartum = ['BabyBumpsCanada', 'BabyBumps','CoronaBumpers', 'PregnancyUK', 'pregnant']
general_depression = ['SuicideWatch', 'depression_help', 'depressed', 'HealthAnxiety', 'AnxietyDepression', 'depression', 'PanicAttack', 'Anxiety']
selected_column = ["id", "subreddit", "selftext"]
query_pregnant_in_general_depression = "postpartum|peripartum|pregnant|pregnancy|intrapartum|antepartum|childbirth|post partum|peri partum|post-partum|peri-partum|ante partum|ante-partum|giving birth|new mom|newborn|new born|baby|maternity|prenatal|postnatal|trimester"
query_depression_in_non_postpartum = "suicide|suicidal|self harm|self-harm|kill myself|killing myself|depression|depressed|depress|anxiety|anxious|panic attack|panic|mental health|therapy|therapist|medication|antidepressant"

non_and_depress_data = pd.read_parquet("data/non_and_general.parquet")
postpartum_data = pd.read_parquet("data/postpartum.parquet")

general_depression_data = non_and_depress_data[
    non_and_depress_data["subreddit"].isin(general_depression) & 
    (non_and_depress_data["selftext"].str.match("[deleted]", na=False) == False)&
    (non_and_depress_data["selftext"].str.match("[removed]", na=False) == False) &
    (non_and_depress_data["selftext"].str.len() > 10) &
    (non_and_depress_data["selftext"].str.contains(query_pregnant_in_general_depression, case=False, na=False) == False)
    ][selected_column]
non_pospartum_data = non_and_depress_data[
    non_and_depress_data["subreddit"].isin(non_postpartum) &
    (non_and_depress_data["selftext"].str.match("[deleted]", na=False) == False)&
    (non_and_depress_data["selftext"].str.match("[removed]", na=False) == False) &
    (non_and_depress_data["selftext"].str.len() > 10) &
    (non_and_depress_data["selftext"].str.contains(query_depression_in_non_postpartum, case=False, na=False) == False)
    ][selected_column]

print(
    f"general depression data: {len(general_depression_data.values)}", 
    f"non postpartum data: {len(non_pospartum_data.values)}", 
    f"postpartum data: {len(postpartum_data.values)}"
    )

n_samples = min(
    len(general_depression_data),
    len(non_pospartum_data),
    len(postpartum_data)
)

print(f"Samples per class: {n_samples}")

general_balanced = general_depression_data.sample(n=n_samples, random_state=42)
normal_balanced = non_pospartum_data.sample(n=n_samples, random_state=42)
ppd_balanced = postpartum_data.copy()

general_balanced['label'] = 1  # General Depression
normal_balanced['label'] = 0   # Normal Pregnancy
ppd_balanced['label'] = 2      # Postpartum Depression

df_balanced = pd.concat([
    normal_balanced, 
    general_balanced, 
    ppd_balanced
], ignore_index=True)

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTotal balanced dataset: {len(df_balanced)}")
print("\nClass distribution:")
print(df_balanced['label'].value_counts().sort_index())

df_balanced.to_parquet("data/balanced_dataset.parquet")
print("\n Saved to data/balanced_dataset.parquet")