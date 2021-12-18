package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqrf computes a QR factorization of a complex M-by-N matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where:
//
//    Q is a M-by-M orthogonal matrix;
//    R is an upper-triangular N-by-N matrix;
//    0 is a (M-N)-by-N zero matrix, if M > N.
func Zgeqrf(m, n int, a *mat.CMatrix, tau, work *mat.CVector, lwork int) (err error) {
	var lquery bool
	var i, ib, iws, k, ldwork, lwkopt, nb, nbmin, nx int

	//     Test the input arguments
	nb = Ilaenv(1, "Zgeqrf", []byte{' '}, m, n, -1, -1)
	lwkopt = n * nb
	work.SetRe(0, float64(lwkopt))
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if lwork < max(1, n) && !lquery {
		err = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Zgeqrf", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	k = min(m, n)
	if k == 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	nx = 0
	iws = n
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Zgeqrf", []byte{' '}, m, n, -1, -1))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = n
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = lwork / ldwork
				nbmin = max(2, Ilaenv(2, "Zgeqrf", []byte{' '}, m, n, -1, -1))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code initially
		for i = 1; i <= k-nx; i += nb {
			ib = min(k-i+1, nb)

			//           Compute the QR factorization of the current block
			//           A(i:m,i:i+ib-1)
			if err = Zgeqr2(m-i+1, ib, a.Off(i-1, i-1), tau.Off(i-1), work); err != nil {
				panic(err)
			}
			if i+ib <= n {
				//              Form the triangular factor of the block reflector
				//              H = H(i) H(i+1) . . . H(i+ib-1)
				Zlarft('F', 'C', m-i+1, ib, a.Off(i-1, i-1), tau.Off(i-1), work.CMatrix(ldwork, opts))

				//              Apply H**H to A(i:m,i+ib:n) from the left
				Zlarfb(Left, ConjTrans, 'F', 'C', m-i+1, n-i-ib+1, ib, a.Off(i-1, i-1), work.CMatrix(ldwork, opts), a.Off(i-1, i+ib-1), work.Off(ib).CMatrix(ldwork, opts))
			}
		}
	} else {
		i = 1
	}

	//     Use unblocked code to factor the last or only block.
	if i <= k {
		if err = Zgeqr2(m-i+1, n-i+1, a.Off(i-1, i-1), tau.Off(i-1), work); err != nil {
			panic(err)
		}
	}

	work.SetRe(0, float64(iws))

	return
}
