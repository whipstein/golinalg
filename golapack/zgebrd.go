package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgebrd reduces a general complex M-by-N matrix A to upper or lower
// bidiagonal form B by a unitary transformation: Q**H * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
func Zgebrd(m, n *int, a *mat.CMatrix, lda *int, d, e *mat.Vector, tauq, taup, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var one complex128
	var i, iinfo, j, ldwrkx, ldwrky, lwkopt, minmn, nb, nbmin, nx, ws int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	nb = maxint(1, Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))
	lwkopt = ((*m) + (*n)) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *m) {
		(*info) = -4
	} else if (*lwork) < maxint(1, *m, *n) && !lquery {
		(*info) = -10
	}
	if (*info) < 0 {
		gltest.Xerbla([]byte("ZGEBRD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	minmn = minint(*m, *n)
	if minmn == 0 {
		work.Set(0, 1)
		return
	}

	ws = maxint(*m, *n)
	ldwrkx = (*m)
	ldwrky = (*n)

	if nb > 1 && nb < minmn {
		//        Set the crossover point NX.
		nx = maxint(nb, Ilaenv(func() *int { y := 3; return &y }(), []byte("ZGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1)))

		//        Determine when to switch from blocked to unblocked code.
		if nx < minmn {
			ws = ((*m) + (*n)) * nb
			if (*lwork) < ws {
				//              Not enough work space for the optimal NB, consider using
				//              a smaller block size.
				nbmin = Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGEBRD"), []byte{' '}, m, n, toPtr(-1), toPtr(-1))
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
		//        Reduce rows and columns i:i+ib-1 to bidiagonal form and return
		//        the matrices X and Y which are needed to update the unreduced
		//        part of the matrix
		Zlabrd(toPtr((*m)-i+1), toPtr((*n)-i+1), &nb, a.Off(i-1, i-1), lda, d.Off(i-1), e.Off(i-1), tauq.Off(i-1), taup.Off(i-1), work.CMatrix(ldwrkx, opts), &ldwrkx, work.CMatrixOff(ldwrkx*nb+1-1, ldwrky, opts), &ldwrky)

		//        Update the trailing submatrix A(i+ib:m,i+ib:n), using
		//        an update of the form  A := A - V*Y**H - X*U**H
		err = goblas.Zgemm(NoTrans, ConjTrans, (*m)-i-nb+1, (*n)-i-nb+1, nb, -one, a.Off(i+nb-1, i-1), *lda, work.CMatrixOff(ldwrkx*nb+nb+1-1, ldwrky, opts), ldwrky, one, a.Off(i+nb-1, i+nb-1), *lda)
		err = goblas.Zgemm(NoTrans, NoTrans, (*m)-i-nb+1, (*n)-i-nb+1, nb, -one, work.CMatrixOff(nb+1-1, ldwrkx, opts), ldwrkx, a.Off(i-1, i+nb-1), *lda, one, a.Off(i+nb-1, i+nb-1), *lda)

		//        Copy diagonal and off-diagonal elements of B back into A
		if (*m) >= (*n) {
			for j = i; j <= i+nb-1; j++ {
				a.SetRe(j-1, j-1, d.Get(j-1))
				a.SetRe(j-1, j+1-1, e.Get(j-1))
			}
		} else {
			for j = i; j <= i+nb-1; j++ {
				a.SetRe(j-1, j-1, d.Get(j-1))
				a.SetRe(j+1-1, j-1, e.Get(j-1))
			}
		}
	}

	//     Use unblocked code to reduce the remainder of the matrix
	Zgebd2(toPtr((*m)-i+1), toPtr((*n)-i+1), a.Off(i-1, i-1), lda, d.Off(i-1), e.Off(i-1), tauq.Off(i-1), taup.Off(i-1), work, &iinfo)
	work.SetRe(0, float64(ws))
}
