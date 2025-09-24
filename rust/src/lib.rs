use ndarray::s;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{Lapack, SVD, Scalar, error::LinalgError};

// unfortunaletely, ndarray-linalg does not provide thin_svd function, though internally it appears to have options for thin svd.
// here we define a thin_svd function that uses the full svd and slice the unitary matrices accordingly.

pub trait ThinSVD {
    type U;
    type VT;
    type Sigma;
    fn thin_svd(
        &self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>), LinalgError>;
}

impl<A, S> ThinSVD for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn thin_svd(
        &self,
        calc_u: bool,
        calc_vt: bool,
    ) -> Result<(Option<Self::U>, Self::Sigma, Option<Self::VT>), LinalgError> {
        let (u, s, vt) = self.svd(calc_u, calc_vt)?;

        let u_thin = u.map(|u| u.slice(s![.., ..s.len()]).to_owned());
        let vt_thin = vt.map(|vt| vt.slice(s![..s.len(), ..]).to_owned());

        Ok((u_thin, s, vt_thin))
    }
}
