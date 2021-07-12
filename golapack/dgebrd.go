package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgebrd reduces a general real M-by-N matrix A to upper or lower
// bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
func Dgebrd(m, n *int, a *mat.Matrix, lda *int, d, e, tauq, taup, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var one float64
	var i, iinfo, j, ldwrkx, ldwrky, lwkopt, minmn, nb, nbmin, nx, ws int
	var err error
	_ = err

	one = 1.0

	//     Test the input parameters
	(*info) = 0
	nb = max(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("DGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
	lwkopt = ((*m) + (*n)) * nb
	work.Set(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < max(1, *m) {
		(*info) = -4
	} else if (*lwork) < max(1, *m, *n) && !lquery {
		(*info) = -10
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("DGEBRD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	minmn = min(*m, *n)
	if minmn == 0 {
		work.Set(0, 1)
		return
	}

	ws = max(*m, *n)
	ldwrkx = (*m)
	ldwrky = (*n)

	if nb > 1 && nb < minmn {
		//        Set the crossover point NX.
		nx = max(nb, Ilaenv(func() *int { y := 3; return &y }(), []byte("DGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))

		//        Determine when to switch from blocked to unblocked code.
		if nx < minmn {
			ws = ((*m) + (*n)) * nb
			if (*lwork) < ws {
				//              Not enough work space for the optimal NB, consider using
				//              a smaller block size.
				nbmin = Ilaenv(func() *int { y := 2; return &y }(), []byte("DGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
				if (*lwork) >= ((*m)+(*n))*nbmin {
					nb = (*lwork) / ((*m) + (*n))
				} else {
					nb = 1
					nx = minmn
				}
			}
		}
	} else {
		nx = minmn
	}

	for i = 1; i <= minmn-nx; i += nb {
		//        Reduce rows and columns i:i+nb-1 to bidiagonal form and return
		//        the matrices X and Y which are needed to update the unreduced
		//        part of the matrix
		Dlabrd(toPtr((*m)-i+1), toPtr((*n)-i+1), &nb, a.Off(i-1, i-1), lda, d.Off(i-1), e.Off(i-1), tauq.Off(i-1), taup.Off(i-1), work.Matrix(ldwrkx, opts), &ldwrkx, work.MatrixOff(ldwrkx*nb, ldwrky, opts), &ldwrky)

		//        Update the trailing submatrix A(i+nb:m,i+nb:n), using an update
		//        of the form  A := A - V*Y**T - X*U**T
		err = goblas.Dgemm(NoTrans, Trans, (*m)-i-nb+1, (*n)-i-nb+1, nb, -one, a.Off(i+nb-1, i-1), work.MatrixOff(ldwrkx*nb+nb, ldwrky, opts), one, a.Off(i+nb-1, i+nb-1))
		err = goblas.Dgemm(NoTrans, NoTrans, (*m)-i-nb+1, (*n)-i-nb+1, nb, -one, work.MatrixOff(nb, ldwrkx, opts), a.Off(i-1, i+nb-1), one, a.Off(i+nb-1, i+nb-1))

		//        Copy diagonal and off-diagonal elements of B back into A
		if (*m) >= (*n) {
			for j = i; j <= i+nb-1; j++ {
				a.Set(j-1, j-1, d.Get(j-1))
				a.Set(j-1, j, e.Get(j-1))
			}
		} else {
			for j = i; j <= i+nb-1; j++ {
				a.Set(j-1, j-1, d.Get(j-1))
				a.Set(j, j-1, e.Get(j-1))
			}
		}
	}

	//     Use unblocked code to reduce the remainder of the matrix
	Dgebd2(toPtr((*m)-i+1), toPtr((*n)-i+1), a.Off(i-1, i-1), lda, d.Off(i-1), e.Off(i-1), tauq.Off(i-1), taup.Off(i-1), work, &iinfo)
	work.Set(0, float64(ws))
}
