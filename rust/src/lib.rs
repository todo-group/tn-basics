pub mod plot;

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2, s};
use ndarray_linalg::{Lapack, SVD, Scalar, error::LinalgError};

// unfortunaletely, ndarray-linalg does not provide thin_svd function, though internally it appears to have options for thin svd.
// also, svd function in ndarray-linalg require flags to calc U,Vt, while we usually use both.
// here we define a full_svd / thin_svd function that uses the full svd and slice the unitary matrices accordingly.

pub trait EasySVD {
    type U;
    type VT;
    type Sigma;
    /// Performs thin SVD.
    ///
    /// # Errors
    /// If underlying `.svd` call fails.
    #[expect(clippy::type_complexity)]
    fn thin_svd(&self) -> Result<(Self::U, Self::Sigma, Self::VT), LinalgError>;
    /// Performs full SVD.
    ///
    /// # Errors
    /// If underlying `.svd` call fails.
    #[expect(clippy::type_complexity)]
    fn full_svd(&self) -> Result<(Self::U, Self::Sigma, Self::VT), LinalgError>;
}

impl<A, S> EasySVD for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type U = Array2<A>;
    type VT = Array2<A>;
    type Sigma = Array1<A::Real>;

    fn full_svd(&self) -> Result<(Self::U, Self::Sigma, Self::VT), LinalgError> {
        let (u, s, vt) = self.svd(true, true)?;
        let u = u.unwrap();
        let vt = vt.unwrap();

        Ok((u, s, vt))
    }

    fn thin_svd(&self) -> Result<(Self::U, Self::Sigma, Self::VT), LinalgError> {
        let (u, s, vt) = self.full_svd()?;
        let u_thin = u.slice_move(s![.., ..s.len()]);
        let vt_thin = vt.slice_move(s![..s.len(), ..]);

        Ok((u_thin, s, vt_thin))
    }
}

pub trait MapStrToAnyhowErr {
    type Ok;
    #[expect(clippy::missing_errors_doc)]
    fn map_str_err(self) -> anyhow::Result<Self::Ok>;
}

impl<T> MapStrToAnyhowErr for Result<T, &'static str> {
    type Ok = T;
    fn map_str_err(self) -> anyhow::Result<<Self as MapStrToAnyhowErr>::Ok> {
        self.map_err(|e| anyhow::anyhow!(e))
    }
}
