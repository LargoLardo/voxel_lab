"""Specialists can learn genes without an all-family model or checkpoint."""
import pytest
import torch

from morphovoxel.checkpointing import initialize_tree_family, save_checkpoint
from morphovoxel.genomes import FAMILY_GENE_NAMES, TREE_FAMILIES, TreeGenome, tree_genome_tensor
from morphovoxel.lab import LabSession
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.state import StateLayout
from morphovoxel.training.family import sample_counterfactual_family_data
from morphovoxel.training.trainer import train


@pytest.mark.parametrize('family', TREE_FAMILIES)
def test_each_specialist_trains_variation_and_live_gene_edits(tmp_path, family):
    base = {
        'runs_root': str(tmp_path), 'device': 'cpu', 'world_size': 12,
        'batch_size': 2, 'iterations': 2, 'rollout_steps': 1, 'persistence_steps': 1,
        'validation_steps': 1, 'validation_every': 2, 'validation_min_steps': 1,
        'validation_recovery_steps': 1, 'validation_min_recovery_steps': 1,
        'validation_fire_seeds': [1], 'validation_random_count': 0,
        'validation_interpolation_steps': 0, 'validation_mutation_count': 0,
        'family_style_seeds': [0], 'transition_source_steps': 2, 'pool_size': 16,
    }
    source = train({
        **{key: value for key, value in base.items() if key != 'family_style_seeds'},
        'run_name': 'specialist', 'model_kind': 'tree_specialist', 'conditional': False,
        'environment_conditioning': False, 'materials': 4, 'hidden_channels': 1,
        'hidden_layers': [4, 4], 'fire_rate': 1., 'tree_genome': {'family': family},
    }, dimensions=3)
    source_checkpoint = source / 'checkpoints/latest.pt'
    for mode in ('variation', 'gene_transition'):
        run = train({
            **base, 'run_name': mode, 'model_kind': 'tree_gene',
            'family_curriculum': mode, 'initialize_from_checkpoint': str(source_checkpoint),
        }, dimensions=3, conditional=True)
        payload = torch.load(run / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        assert payload['config']['tree_genome']['family'] == family
        assert payload['config']['hidden_layers'] == [4, 4]
        assert payload['config']['hidden_channels'] == 1
        assert payload['metadata']['model_kind'] == 'tree_gene'
        assert all(key.startswith('update.') for key in payload['model'])
        pool = payload['pool']
        assert len(pool['states']) == 16  # Only eight gene conditions, not four families.
        assert set(pool['genomes'][:, :4].argmax(1).tolist()) == {TREE_FAMILIES.index(family)}
        assert len(set(pool['condition_ids'].tolist())) == len(FAMILY_GENE_NAMES)
        panel = payload['validation']['validation_panel']
        assert panel and {case['genome']['family'] for case in panel} == {family}
        if mode == 'gene_transition':
            assert len(panel) == 32
            assert all(case['source_genome']['family'] == family for case in panel)
        lab = LabSession.from_run(run, 'cpu', 'latest.pt')
        assert isinstance(lab.model, NeuralCA3D)
        assert lab.summary()['fixed_tree_family'] == family
        lab.advance(2)
        before = lab.state.clone()
        changed = lab.active_tree_genome.with_values({'height': .5})
        lab.set_tree_genome(changed.to_dict(), live_remodel=True)
        torch.testing.assert_close(lab.state, before)
        assert lab.active_tree_genome == changed
        lab.advance(1)
        with pytest.raises(ValueError, match='specialist tree type'):
            lab.set_tree_genome(TreeGenome(family=TREE_FAMILIES[(TREE_FAMILIES.index(family) + 1) % 4]).to_dict())
    if family == 'branching':
        handoff = train({
            **base, 'run_name': 'handoff', 'model_kind': 'tree_gene',
            'family_curriculum': 'gene_transition',
            'initialize_from_checkpoint': str(tmp_path / 'variation/checkpoints/latest.pt'),
        }, dimensions=3, conditional=True)
        resumed = train({
            **base, 'run_name': 'resumed', 'model_kind': 'tree_gene',
            'family_curriculum': 'gene_transition', 'iterations': 1,
            'resume': str(handoff / 'checkpoints/latest.pt'),
        }, dimensions=3, conditional=True)
        assert torch.load(resumed / 'checkpoints/latest.pt', weights_only=False)['step'] == 3


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='Apple GPU unavailable'))])
def test_gene_conversion_preserves_specialist_rule_before_learning(tmp_path, device):
    layout = StateLayout(4, 1)
    specialist = NeuralCA3D(layout.channels, hidden_layers=[8, 4], context_channels=12).to(device)
    genes = NeuralCA3D(layout.channels, genome_size=TreeGenome.model_size(), hidden_layers=[8, 4], context_channels=12).to(device)
    path = tmp_path / 'source.pt'
    save_checkpoint(path, specialist, config={'model_kind': 'tree_specialist', 'tree_genome': {'family': 'weeping'}})
    initialize_tree_family(path, genes)
    state = torch.rand(4, layout.channels, 4, 4, 4, device=device)
    context = torch.rand(4, 12, 4, 4, 4, device=device)
    fire = torch.ones_like(state[:, :1])
    genome = tree_genome_tensor([TreeGenome.random(seed, family='weeping') for seed in range(4)], device=device)
    torch.testing.assert_close(specialist(state, context=context, fire_mask=fire), genes(state, genome, context, fire))
    genes(state, genome, context, fire).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in genes.parameters())
    assert genes.update[0].weight.grad[:, layout.channels * 5:layout.channels * 5 + TreeGenome.model_size()].abs().sum() > 0


def test_gene_training_rejects_wrong_families_and_family_curricula(tmp_path):
    layout = StateLayout(4, 1)
    path = tmp_path / 'conifer.pt'
    save_checkpoint(path, NeuralCA3D(layout.channels, 4), config={
        'model_kind': 'tree_specialist', 'tree_genome': {'family': 'conifer'},
    })
    with pytest.raises(ValueError, match='specializes in conifer'):
        train({'model_kind': 'tree_gene', 'initialize_from_checkpoint': str(path),
               'tree_genome': {'family': 'weeping'}}, dimensions=3, conditional=True)
    for mode in ('full', 'basics', 'transition'):
        with pytest.raises(ValueError, match='variation or gene_transition'):
            train({'model_kind': 'tree_gene', 'family_curriculum': mode}, dimensions=3, conditional=True)
    with pytest.raises(ValueError, match='selected families'):
        sample_counterfactual_family_data(1, 12, 0, families=['conifer'], condition_ids=[0])
