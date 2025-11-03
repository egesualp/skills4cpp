# test_real_data.py
import pandas as pd
import torch
from skill_mapping.v1.data import PillarDataset, SkillDataset, build_job_text, build_skill_text


def load_data(
    esco_path: str = "data/processed/master_datasets/master_complete_hierarchy_w_occ.csv",
    decorte_path: str = "data/title_pairs_desc/decorte_train_pairs.csv",
):
    # Read ESCO master
    esco_df = pd.read_csv(esco_path)

    # Read DECORTE
    decorte_df = pd.read_csv(decorte_path)

    # Make sure DECORTE has a stable index name, because PillarDataset/SKillDataset
    # groupby(decorte_df.index.name or "index")
    if decorte_df.index.name is None:
        decorte_df = decorte_df.reset_index().set_index("index")

    return esco_df, decorte_df


def sanity_print_helpers(esco_df, decorte_df):
    print("=== Basic shapes ===")
    print("ESCO df shape:", esco_df.shape)
    print("DECORTE df shape:", decorte_df.shape)

    print("\n=== Column check (first 20 cols each) ===")
    print("ESCO cols:", list(esco_df.columns)[:20])
    print("DECORTE cols:", list(decorte_df.columns)[:20])

    # Show first ESCO row job text / skill text
    print("\n=== build_job_text / build_skill_text sanity ===")
    esco_row = esco_df.iloc[0]
    decorte_row = decorte_df.iloc[0]

    print("ESCO build_job_text(title):",
          build_job_text(esco_row, text_fields="title"))
    print("ESCO build_job_text(title+desc):",
          build_job_text(esco_row, text_fields="title+desc"))
    print("ESCO build_job_text(title+desc+alt):",
          build_job_text(esco_row, text_fields="title+desc+alt"))

    print("\nDECORTE build_job_text(title):",
          build_job_text(decorte_row, text_fields="title"))
    print("DECORTE build_job_text(title+desc):",
          build_job_text(decorte_row, text_fields="title+desc"))

    print("\nESCO build_skill_text(title):",
          build_skill_text(esco_row, text_fields="title"))
    print("ESCO build_skill_text(title+desc):",
          build_skill_text(esco_row, text_fields="title+desc"))
    print("ESCO build_skill_text(title+desc+alt):",
          build_skill_text(esco_row, text_fields="title+desc+alt"))
    print()


def test_pillar_dataset(esco_df, decorte_df):
    print("=== PillarDataset test ===")

    # pick a hierarchy level to predict. level0 is broadest.
    hier_level = 0

    pillar_esco = PillarDataset(
        esco_df=esco_df,
        decorte_df=None,
        hier_level=hier_level,
        source="esco",
        text_fields="title+desc",  # you can also try "title"
    )

    pillar_all = PillarDataset(
        esco_df=esco_df,
        decorte_df=decorte_df,
        hier_level=hier_level,
        source="all",              # ESCO + DECORTE
        text_fields="title+desc",
    )

    print("ESCO-only PillarDataset len:", len(pillar_esco))
    print("ESCO-only num_classes():", pillar_esco.num_classes())

    # show vocab size and a few labels
    vocab_esco = pillar_esco.categories_vocab()
    print("ESCO-only categories_vocab size:", len(vocab_esco))
    print("Sample of categories_vocab (first 10):",
          {k: vocab_esco[k] for k in list(vocab_esco.keys())[:10]})

    print("\nALL-source PillarDataset len:", len(pillar_all))
    print("ALL-source num_classes():", pillar_all.num_classes())
    vocab_all = pillar_all.categories_vocab()
    print("ALL-source categories_vocab size:", len(vocab_all))
    print("Sample of categories_vocab (first 10):",
          {k: vocab_all[k] for k in list(vocab_all.keys())[:10]})

    # Look at a couple samples
    def show_sample(ds, idx):
        ex = ds[idx]
        print(f"\nSample ds[{idx}] text:")
        print(ex["text"])
        print("target y shape:", tuple(ex["y"].shape))
        print("target y nonzero idx:",
              torch.nonzero(ex["y"]).view(-1).tolist())
        print("meta:", ex["meta"])

    show_sample(pillar_esco, 0)
    if len(pillar_all) > 1:
        show_sample(pillar_all, 1)

    print()


def test_skill_dataset(esco_df, decorte_df):
    print("=== SkillDataset test ===")

    # We need to pick one concrete category_value that exists in level0_label
    # We'll just grab the first non-null level0_label from ESCO.
    first_cat = (
        esco_df["level0_label"]
        .dropna()
        .astype(str)
        .iloc[0]
    )

    print("Chosen category_value for SkillDataset:", first_cat)

    skill_ds = SkillDataset(
        esco_df=esco_df,
        decorte_df=decorte_df,
        hier_level=0,                # must match the column we sampled from
        category_value=first_cat,
        source="all",                # train from ESCO + DECORTE
        text_fields="title+desc",    # same idea as Stage 1
    )

    print("SkillDataset len:", len(skill_ds))
    print("SkillDataset num_skills():", skill_ds.num_skills())

    vocab_skills = skill_ds.skills_vocab()
    print("SkillDataset skills_vocab size:", len(vocab_skills))
    print("Sample skills_vocab (first 10):",
          {k: vocab_skills[k] for k in list(vocab_skills.keys())[:10]})

    # Inspect one sample
    if len(skill_ds) > 0:
        ex = skill_ds[0]
        print("\nSample skill_ds[0] text:")
        print(ex["text"])
        print("target y shape:", tuple(ex["y"].shape))
        print("target y nonzero idx:",
              torch.nonzero(ex["y"]).view(-1).tolist())
        print("meta:", ex["meta"])
    else:
        print("No samples matched that category.")

    print()


def main():
    esco_df, decorte_df = load_data()

    print("=== Loaded dataframes ===")
    print("esco_df head():")
    print(esco_df.head(3))
    print("\ndecorte_df head():")
    print(decorte_df.head(3))
    print()

    # 1. Sanity: helper text builders
    sanity_print_helpers(esco_df, decorte_df)

    # 2. PillarDataset (Stage 1)
    test_pillar_dataset(esco_df, decorte_df)

    # 3. SkillDataset (Stage 2)
    test_skill_dataset(esco_df, decorte_df)

    print("Done.")


if __name__ == "__main__":
    main()
