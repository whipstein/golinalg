package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpptri computes the inverse of a complex Hermitian positive definite
// matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
// computed by ZPPTRF.
func Zpptri(uplo byte, n *int, ap *mat.CVector, info *int) {
	var upper bool
	var ajj, one float64
	var j, jc, jj, jjn int

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
		gltest.Xerbla([]byte("ZPPTRI"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Invert the triangular Cholesky factor U or L.
	Ztptri(uplo, 'N', n, ap, info)
	if (*info) > 0 {
		return
	}
	if upper {
		//        Compute the product inv(U) * inv(U)**H.
		jj = 0
		for j = 1; j <= (*n); j++ {
			jc = jj + 1
			jj = jj + j
			if j > 1 {
				goblas.Zhpr(Upper, toPtr(j-1), &one, ap.Off(jc-1), func() *int { y := 1; return &y }(), ap)
			}
			ajj = ap.GetRe(jj - 1)
			goblas.Zdscal(&j, &ajj, ap.Off(jc-1), func() *int { y := 1; return &y }())
		}

	} else {
		//        Compute the product inv(L)**H * inv(L).
		jj = 1
		for j = 1; j <= (*n); j++ {
			jjn = jj + (*n) - j + 1
			ap.SetRe(jj-1, real(goblas.Zdotc(toPtr((*n)-j+1), ap.Off(jj-1), func() *int { y := 1; return &y }(), ap.Off(jj-1), func() *int { y := 1; return &y }())))
			if j < (*n) {
				goblas.Ztpmv(Lower, ConjTrans, NonUnit, toPtr((*n)-j), ap.Off(jjn-1), ap.Off(jj+1-1), func() *int { y := 1; return &y }())
			}
			jj = jjn
		}
	}
}
