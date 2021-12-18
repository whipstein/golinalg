package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpbstf computes a split Cholesky factorization of a real
// symmetric positive definite band matrix A.
//
// This routine is designed to be used in conjunction with DSBGST.
//
// The factorization has the form  A = S**T*S  where S is a band matrix
// of the same bandwidth as A and the following structure:
//
//   S = ( U    )
//       ( M  L )
//
// where U is upper triangular of order m = (n+kd)/2, and L is lower
// triangular of order n-m.
func Dpbstf(uplo mat.MatUplo, n, kd int, ab *mat.Matrix) (info int, err error) {
	var upper bool
	var ajj, one, zero float64
	var j, kld, km, m int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	}
	if err != nil {
		gltest.Xerbla2("Dpbstf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	kld = max(1, ab.Rows-1)

	//     Set the splitting point m.
	m = (n + kd) / 2

	if upper {
		//        Factorize A(m+1:n,m+1:n) as L**T*L, and update A(1:m,1:m).
		for j = n; j >= m+1; j-- {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.Get(kd, j-1)
			if ajj <= zero {
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.Set(kd, j-1, ajj)
			km = min(j-1, kd)

			//           Compute elements j-km:j-1 of the j-th column and update the
			//           the leading submatrix within the band.
			ab.Off(kd+1-km-1, j-1).Vector().Scal(km, one/ajj, 1)
			if err = ab.OffIdx(kd+(j-km-1)*ab.Rows).UpdateRows(kld).Syr(Upper, km, -one, ab.Off(kd+1-km-1, j-1).Vector(), 1); err != nil {
				panic(err)
			}
		}

		//        Factorize the updated submatrix A(1:m,1:m) as U**T*U.
		for j = 1; j <= m; j++ {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.Get(kd, j-1)
			if ajj <= zero {
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.Set(kd, j-1, ajj)
			km = min(kd, m-j)

			//           Compute elements j+1:j+km of the j-th row and update the
			//           trailing submatrix within the band.
			if km > 0 {
				ab.Off(kd-1, j).Vector().Scal(km, one/ajj, kld)
				if err = ab.OffIdx(kd+(j)*ab.Rows).UpdateRows(kld).Syr(Upper, km, -one, ab.Off(kd-1, j).Vector(), kld); err != nil {
					panic(err)
				}
			}
		}
	} else {
		//        Factorize A(m+1:n,m+1:n) as L**T*L, and update A(1:m,1:m).
		for j = n; j >= m+1; j-- {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.Get(0, j-1)
			if ajj <= zero {
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.Set(0, j-1, ajj)
			km = min(j-1, kd)

			//           Compute elements j-km:j-1 of the j-th row and update the
			//           trailing submatrix within the band.
			ab.Off(km, j-km-1).Vector().Scal(km, one/ajj, kld)
			if err = ab.OffIdx(0+(j-km-1)*ab.Rows).UpdateRows(kld).Syr(Lower, km, -one, ab.Off(km, j-km-1).Vector(), kld); err != nil {
				panic(err)
			}
		}

		//        Factorize the updated submatrix A(1:m,1:m) as U**T*U.
		for j = 1; j <= m; j++ {
			//           Compute s(j,j) and test for non-positive-definiteness.
			ajj = ab.Get(0, j-1)
			if ajj <= zero {
				goto label50
			}
			ajj = math.Sqrt(ajj)
			ab.Set(0, j-1, ajj)
			km = min(kd, m-j)

			//           Compute elements j+1:j+km of the j-th column and update the
			//           trailing submatrix within the band.
			if km > 0 {
				ab.Off(1, j-1).Vector().Scal(km, one/ajj, 1)
				if err = ab.OffIdx(0+(j)*ab.Rows).UpdateRows(kld).Syr(Lower, km, -one, ab.Off(1, j-1).Vector(), 1); err != nil {
					panic(err)
				}
			}
		}
	}
	return

label50:
	;
	info = j

	return
}
