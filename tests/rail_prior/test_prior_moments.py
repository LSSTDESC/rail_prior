import qp
import numpy as np
import rail_prior


def make_qp_ens(file):
    zs = file['zs']
    nzs = file['pzs']
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz/2, zs[-1] + dz/2)
    q = qp.Ensemble(qp.hist, data={"bins":zs_edges, "pdfs":nzs})
    return q


def make_prior():
    file = np.load('tests/rail_prior/dummy.npz')
    ens = make_qp_ens(file)
    return rail_prior.PriorMoments(ens)


def test_prior():
    projector = make_prior()
    prior = rail_prior.get_prior()
    assert prior is not None


def test_sample_prior():
    prior = make_prior()
    nz = prior.sample_prior()
    assert len(nz) == len(prior.nz_mean)


def test_model():
    prior = make_prior()
    shift = prior.sample_prior()
    input = np.array([prior.z, prior.nz_mean])
    output = prior.evaluate_model(input, shift)
    assert (prior.z == output[0]).all()
    assert len(output[1]) == len(prior.nz_mean)
