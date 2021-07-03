package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Ztpt01 computes the residual for a triangular matrix A times its
// inverse when A is stored in packed format:
//    RESID = norm(A*AINV - I) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func Ztpt01(uplo, diag byte, n *int, ap, ainvp *mat.CVector, rcond *float64, rwork *mat.Vector, resid *float64) {
	var unitd bool
	var ainvnm, anorm, eps, one, zero float64
	var j, jc int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if (*n) <= 0 {
		(*rcond) = one
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlantp('1', uplo, diag, n, ap, rwork)
	ainvnm = golapack.Zlantp('1', uplo, diag, n, ainvp, rwork)
	if anorm <= zero || ainvnm <= zero {
		(*rcond) = zero
		(*resid) = one / eps
		return
	}
	(*rcond) = (one / anorm) / ainvnm

	//     Compute A * AINV, overwriting AINV.
	unitd = diag == 'U'
	if uplo == 'U' {
		jc = 1
		for j = 1; j <= (*n); j++ {
			if unitd {
				ainvp.SetRe(jc+j-1-1, one)
			}

			//           Form the j-th column of A*AINV.
			err = goblas.Ztpmv(Upper, NoTrans, mat.DiagByte(diag), j, ap, ainvp.Off(jc-1), 1)

			//           Subtract 1 from the diagonal to form A*AINV - I.
			ainvp.Set(jc+j-1-1, ainvp.Get(jc+j-1-1)-complex(one, 0))
			jc = jc + j
		}
	} else {
		jc = 1
		for j = 1; j <= (*n); j++ {
			if unitd {
				ainvp.SetRe(jc-1, one)
			}

			//           Form the j-th column of A*AINV.
			err = goblas.Ztpmv(Lower, NoTrans, mat.DiagByte(diag), (*n)-j+1, ap.Off(jc-1), ainvp.Off(jc-1), 1)

			//           Subtract 1 from the diagonal to form A*AINV - I.
			ainvp.Set(jc-1, ainvp.Get(jc-1)-complex(one, 0))
			jc = jc + (*n) - j + 1
		}
	}

	//     Compute norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS)
	(*resid) = golapack.Zlantp('1', uplo, 'N', n, ainvp, rwork)

	(*resid) = (((*resid) * (*rcond)) / float64(*n)) / eps
}
