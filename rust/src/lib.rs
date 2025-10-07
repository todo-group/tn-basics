pub mod plot;

#[cfg(feature = "gtk4")]
pub mod gtk_plot;

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{JobSvd, Lapack, SVDDC, Scalar, error::LinalgError};
extern crate blas_src as _;

// ~~ unfortunaletely, ndarray-linalg does not provide thin_svd function, though internally it appears to have options for thin svd. ~~ SVDDC has options for full/thin svd, but it requires to provide JobSvd flag.
// also, svd function in ndarray-linalg require flags to calc U,Vt, while we usually use both.
// here we define a full_svd / thin_svd function for convenience.

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
        let (u, s, vt) = self.svddc(JobSvd::All)?;
        let u = u.unwrap();
        let vt = vt.unwrap();

        Ok((u, s, vt))
    }

    fn thin_svd(&self) -> Result<(Self::U, Self::Sigma, Self::VT), LinalgError> {
        let (u, s, vt) = self.svddc(JobSvd::Some)?;
        let u = u.unwrap();
        let vt = vt.unwrap();

        Ok((u, s, vt))
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
