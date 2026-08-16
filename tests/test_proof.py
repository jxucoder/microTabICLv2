from pathlib import Path

from microtabiclv2.config import ModelConfig
from microtabiclv2.model import MicroTabICLv2
from microtabiclv2.proof import rule_switching_proof
from microtabiclv2.train import save_checkpoint


def test_rule_switching_proof_writes_svg(tmp_path: Path):
    config = ModelConfig(
        embed_dim=16,
        col_blocks=1,
        row_blocks=1,
        icl_blocks=1,
        col_heads=4,
        row_heads=4,
        icl_heads=4,
        inducing_tokens=4,
        row_cls_tokens=2,
        qass_hidden_dim=8,
    )
    checkpoint = tmp_path / "tiny.pt"
    output = tmp_path / "proof.svg"
    save_checkpoint(checkpoint, MicroTabICLv2(config))
    scores = rule_switching_proof(
        checkpoint,
        output,
        device="cpu",
        repeats=1,
        context_rows=8,
        grid_size=9,
    )
    assert set(("vertical", "diagonal", "horizontal", "anti-diagonal")) <= scores.keys()
    assert output.read_text().startswith('<svg xmlns="http://www.w3.org/2000/svg"')
