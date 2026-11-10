import json
from pathlib import Path

import numpy as np
import torch

from vivivit import build_model
from vivivit.bispectrum import MuResolvedBispectrumGPU
from vivivit.data import Lightcones, simulation_files, split_dataset
from vivivit.loss import reconstruction_loss
from vivivit.power_spectrum import PkEstimator, ShellPkEstimator
from vivivit.taper import taper_field, tukey_window
from vivivit.postprocess import triangle_families, collapse_orientations

ROOT = Path(__file__).resolve().parents[1]


def test_production_checkpoint_shapes():
    for record in json.loads((ROOT/'checkpoint_manifest.json').read_text()):
        with torch.device('meta'):
            model = build_model(record['config'])
        assert {k: list(v.shape) for k, v in model.state_dict().items()} == record['state_shapes']
        assert sum(p.numel() for p in model.parameters()) == record['num_state_elements']


def test_normalization_split_and_auxiliary_file_exclusion(tmp_path):
    for i in range(20):
        torch.save((torch.ones(1, 8, 8, 8)*4, torch.ones(1, 8, 8, 8)*8), tmp_path/f'sim_{i:04d}.pt')
    torch.save({}, tmp_path/'sim_0001.pk.pt')
    files = simulation_files(tmp_path, {19})
    dataset = Lightcones(files, dict(input_mean=2, input_std=2, target_mean=4, target_std=4))
    x, y = dataset[0]
    assert torch.all(x == 1) and torch.all(y == 1)
    train, validation = split_dataset(dataset)
    assert len(files) == 19
    assert set(train.indices).isdisjoint(validation.indices)
    assert len(train)+len(validation) == 19


def test_power_amplitude_and_cross_correlation():
    torch.manual_seed(8)
    field = torch.randn(16, 16, 16)
    result = PkEstimator(64).compute(field, 2*field, MAS=('CIC', 'CIC'))[0]
    torch.testing.assert_close(result[:, 2], 4*result[:, 1])
    torch.testing.assert_close(result[:, 3], 2*result[:, 1])
    assert result.shape == (7, 5)


def test_bispectrum_cubic_scaling():
    estimator = MuResolvedBispectrumGPU(32, 8, 2, 'cpu')
    field = np.random.default_rng(4).normal(size=(8, 8, 8))
    estimator.read_real(field, 'CIC')
    a = estimator.Bk_mu(Nbins=3)
    estimator.read_real(2*field, 'CIC')
    b = estimator.Bk_mu(Nbins=3)
    np.testing.assert_allclose(b['B'], 8*a['B'], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(a['counts'], b['counts'])


def test_loss_constant_offset_and_gradient():
    prediction = torch.ones(1, 1, 4, 4, 4, requires_grad=True)
    target = torch.zeros_like(prediction)
    value = reconstruction_loss(prediction, target)
    assert value.item() == 1.
    value.backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 1/64))


def test_bispectrum_orientation_and_family_definitions():
    values = np.arange(8.).reshape(1, 1, 2, 2, 2)
    counts = np.ones_like(values)
    cells = np.ones((2, 2, 2), dtype=bool)
    np.testing.assert_array_equal(collapse_orientations(values, counts, cells), [[3.5]])
    masks = triangle_families(np.array([[0, 3, 3], [1, 1, 3], [3, 3, 3]]), np.array([1., 2.2, 3., 4.]))
    assert masks['squeezed'].tolist() == [True, False, False]
    assert masks['folded'].tolist() == [False, True, False]
    assert masks['equilateral'].tolist() == [False, False, True]


def test_tukey_taper_and_shell_power():
    from scipy.signal.windows import tukey
    for n, alpha in [(128, .5), (37, .25), (16, 1.)]:
        np.testing.assert_allclose(tukey_window(n, alpha), tukey(n, alpha), atol=1e-12)
    field = np.random.default_rng(3).normal(size=(16, 16, 16))+5
    tapered, w2, w3 = taper_field(field, .5)
    w = tukey_window(16, .5)
    assert abs(tapered.mean()) < 1e-12 and np.isclose(w2, np.mean(w**2)) and np.isclose(w3, np.mean(w**3))
    np.testing.assert_allclose(taper_field(field, 0)[0], field-field.mean())
    result = ShellPkEstimator(64).compute(torch.tensor(tapered), 2*torch.tensor(tapered))[0]
    torch.testing.assert_close(result[:, 2], 4*result[:, 1])
    torch.testing.assert_close(result[:, 3], 2*result[:, 1])
    assert result.shape == (13, 5)
    assert float(result[:, 4].sum()) == (16**3-1)/2+3.5


if __name__ == '__main__':
    from tempfile import TemporaryDirectory
    tests = [test_production_checkpoint_shapes, test_power_amplitude_and_cross_correlation, test_tukey_taper_and_shell_power, test_bispectrum_cubic_scaling, test_loss_constant_offset_and_gradient, test_bispectrum_orientation_and_family_definitions]
    torch.set_num_threads(2)
    for test in tests:
        test()
        print(test.__name__+': passed')
    with TemporaryDirectory() as directory:
        test_normalization_split_and_auxiliary_file_exclusion(Path(directory))
    print('7 tests passed')
