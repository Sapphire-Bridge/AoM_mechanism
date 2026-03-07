from pathlib import Path

from aom.reporting import generate_results_report


def test_generate_results_report_smoke(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "aom_eval.csv").write_text(
        "\n".join(
            [
                "aom_composite,model,arch,seed,bootstrap_n,disamb_accuracy,cf_shift_direction_accuracy,coh_constraint_accuracy,disamb_n_pairs_total,cf_n_items_total,coh_n_items_total",
                "0.9,gpt2,gpt2,0,1000,0.8,0.7,0.6,52,120,40",
                "0.85,gpt2,gpt2,1,1000,0.8,0.7,0.6,52,120,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (results_dir / "induction_gpt2.csv").write_text(
        "\n".join(
            [
                "model,arch,baseline_mode,base_len,repeats,seed,device,torch_dtype,layer,head,induction_advantage",
                "gpt2,gpt2,shuffle,64,2,0,cpu,float16,0,0,0.01",
                "gpt2,gpt2,shuffle,64,2,0,cpu,float16,0,1,0.02",
                "gpt2,gpt2,shuffle,64,2,0,cpu,float16,1,0,0.03",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (results_dir / "logit_lens_trace.csv").write_text(
        "\n".join(
            [
                "model,arch,lens,state_index,block_index,logit_diff,final_logit_diff",
                "gpt2,gpt2,auto,0,0,1.0,2.0",
                "gpt2,gpt2,auto,1,1,1.5,2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = generate_results_report(results_dir, include_file_inventory=True)

    assert "Results report" in report
    assert "## Models observed" in report
    assert "gpt2" in report
    assert "AoM evaluation summary" in report
    assert "Induction summary" in report
    assert "Logit lens summary" in report

