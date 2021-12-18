package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgebrd reduces a general complex M-by-N matrix A to upper or lower
// bidiagonal form B by a unitary transformation: Q**H * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
func Zgebrd(m, n int, a *mat.CMatrix, d, e *mat.Vector, tauq, taup, work *mat.CVector, lwork int) (err error) {
	var lquery bool
	var one complex128
	var i, j, ldwrkx, ldwrky, lwkopt, minmn, nb, nbmin, nx, ws int

	one = (1.0 + 0.0*1i)

	//     Test the input parameters
	nb = max(1, Ilaenv(1, "Zgebrd", []byte{' '}, m, n, -1, -1))
	lwkopt = (m + n) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if lwork < max(1, m, n) && !lquery {
		err = fmt.Errorf("lwork < max(1, m, n) && !lquery: lwork=%v, m=%v, n=%v", lwork, m, n)
	}
	if err != nil {
		gltest.Xerbla2("Zgebrd", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	minmn = min(m, n)
	if minmn == 0 {
		work.Set(0, 1)
		return
	}

	ws = max(m, n)
	ldwrkx = m
	ldwrky = n

	if nb > 1 && nb < minmn {
		//        Set the crossover point NX.
		nx = max(nb, Ilaenv(3, "Zgebrd", []byte{' '}, m, n, -1, -1))

		//        Determine when to switch from blocked to unblocked code.
		if nx < minmn {
			ws = (m + n) * nb
			if lwork < ws {
				//              Not enough work space for the optimal NB, consider using
				//              a smaller block size.
				nbmin = Ilaenv(2, "Zgebrd", []byte{' '}, m, n, -1, -1)
				if lwork >= (m+n)*nbmin {
					nb = lwork / (m + n)
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
		Zlabrd(m-i+1, n-i+1, nb, a.Off(i-1, i-1), d.Off(i-1), e.Off(i-1), tauq.Off(i-1), taup.Off(i-1), work.CMatrix(ldwrkx, opts), work.Off(ldwrkx*nb).CMatrix(ldwrky, opts))

		//        Update the trailing submatrix A(i+ib:m,i+ib:n), using
		//        an update of the form  A := A - V*Y**H - X*U**H
		if err = a.Off(i+nb-1, i+nb-1).Gemm(NoTrans, ConjTrans, m-i-nb+1, n-i-nb+1, nb, -one, a.Off(i+nb-1, i-1), work.Off(ldwrkx*nb+nb).CMatrix(ldwrky, opts), one); err != nil {
			panic(err)
		}
		if err = a.Off(i+nb-1, i+nb-1).Gemm(NoTrans, NoTrans, m-i-nb+1, n-i-nb+1, nb, -one, work.Off(nb).CMatrix(ldwrkx, opts), a.Off(i-1, i+nb-1), one); err != nil {
			panic(err)
		}

		//        Copy diagonal and off-diagonal elements of B back into A
		if m >= n {
			for j = i; j <= i+nb-1; j++ {
				a.SetRe(j-1, j-1, d.Get(j-1))
				a.SetRe(j-1, j, e.Get(j-1))
			}
		} else {
			for j = i; j <= i+nb-1; j++ {
				a.SetRe(j-1, j-1, d.Get(j-1))
				a.SetRe(j, j-1, e.Get(j-1))
			}
		}
	}

	//     Use unblocked code to reduce the remainder of the matrix
	if err = Zgebd2(m-i+1, n-i+1, a.Off(i-1, i-1), d.Off(i-1), e.Off(i-1), tauq.Off(i-1), taup.Off(i-1), work); err != nil {
		panic(err)
	}
	work.SetRe(0, float64(ws))

	return
}
