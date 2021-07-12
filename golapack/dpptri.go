package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpptri computes the inverse of a real symmetric positive definite
// matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
// computed by DPPTRF.
func Dpptri(uplo byte, n *int, ap *mat.Vector, info *int) {
	var upper bool
	var ajj, one float64
	var j, jc, jj, jjn int
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DPPTRI"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Invert the triangular Cholesky factor U or L.
	Dtptri(uplo, 'N', n, ap, info)
	if (*info) > 0 {
		return
	}

	if upper {
		//        Compute the product inv(U) * inv(U)**T.
		jj = 0
		for j = 1; j <= (*n); j++ {
			jc = jj + 1
			jj = jj + j
			if j > 1 {
				err = goblas.Dspr(mat.Upper, j-1, one, ap.Off(jc-1, 1), ap)
			}
			ajj = ap.Get(jj - 1)
			goblas.Dscal(j, ajj, ap.Off(jc-1, 1))
		}

	} else {
		//        Compute the product inv(L)**T * inv(L).
		jj = 1
		for j = 1; j <= (*n); j++ {
			jjn = jj + (*n) - j + 1
			ap.Set(jj-1, goblas.Ddot((*n)-j+1, ap.Off(jj-1, 1), ap.Off(jj-1, 1)))
			if j < (*n) {
				err = goblas.Dtpmv(mat.Lower, mat.Trans, mat.NonUnit, (*n)-j, ap.Off(jjn-1), ap.Off(jj, 1))
			}
			jj = jjn
		}
	}
}
