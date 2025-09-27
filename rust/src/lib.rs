use ndarray::s;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{Lapack, SVD, Scalar, error::LinalgError};

// unfortunaletely, ndarray-linalg does not provide thin_svd function, though internally it appears to have options for thin svd.
// also, svd function in ndarray-linalg require flags to calc U,Vt, while we usually use both.
// here we define a full_svd / thin_svd function that uses the full svd and slice the unitary matrices accordingly.

pub trait EasySVD {
    type U;
    type VT;
    type Sigma;
    fn thin_svd(&self) -> Result<(Self::U, Self::Sigma, Self::VT), LinalgError>;
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

        let u_thin = u.slice(s![.., ..s.len()]).into_owned();
        let vt_thin = vt.slice(s![..s.len(), ..]).into_owned();

        Ok((u_thin, s, vt_thin))
    }
}

pub trait MapStrToAnyhowErr {
    type Ok;
    fn map_str_err(self) -> Result<Self::Ok, anyhow::Error>;
}

impl<T> MapStrToAnyhowErr for Result<T, &'static str> {
    type Ok = T;
    fn map_str_err(self) -> Result<<Self as MapStrToAnyhowErr>::Ok, anyhow::Error> {
        self.map_err(|e| anyhow::anyhow!(e))
    }
}
