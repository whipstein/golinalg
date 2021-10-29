package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelqf computes an LQ factorization of a complex M-by-N matrix A:
//
//    A = ( L 0 ) *  Q
//
// where:
//
//    Q is a N-by-N orthogonal matrix;
//    L is an lower-triangular M-by-M matrix;
//    0 is a M-by-(N-M) zero matrix, if M < N.
func Zgelqf(m, n int, a *mat.CMatrix, tau, work *mat.CVector, lwork int) (err error) {
	var lquery bool
	var i, ib, iws, k, ldwork, lwkopt, nb, nbmin, nx int

	//     Test the input arguments
	nb = Ilaenv(1, "Zgelqf", []byte{' '}, m, n, -1, -1)
	lwkopt = m * nb
	work.SetRe(0, float64(lwkopt))
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if lwork < max(1, m) && !lquery {
		err = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=%v, m=%v, lquery=%v", lwork, m, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Zgelqf", err)
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
	iws = m
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Zgelqf", []byte{' '}, m, n, -1, -1))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = m
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = lwork / ldwork
				nbmin = max(2, Ilaenv(2, "Zgelqf", []byte{' '}, m, n, -1, -1))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code initially
		for i = 1; i <= k-nx; i += nb {
			ib = min(k-i+1, nb)

			//           Compute the LQ factorization of the current block
			//           A(i:i+ib-1,i:n)
			if err = Zgelq2(ib, n-i+1, a.Off(i-1, i-1), tau.Off(i-1), work); err != nil {
				panic(err)
			}
			if i+ib <= m {
				//              Form the triangular factor of the block reflector
				//              H = H(i) H(i+1) . . . H(i+ib-1)
				Zlarft('F', 'R', n-i+1, ib, a.Off(i-1, i-1), tau.Off(i-1), work.CMatrix(ldwork, opts))

				//              Apply H to A(i+ib:m,i:n) from the right
				Zlarfb(Right, NoTrans, 'F', 'R', m-i-ib+1, n-i+1, ib, a.Off(i-1, i-1), work.CMatrix(ldwork, opts), a.Off(i+ib-1, i-1), work.CMatrixOff(ib, ldwork, opts))
			}
		}
	} else {
		i = 1
	}

	//     Use unblocked code to factor the last or only block.
	if i <= k {
		if err = Zgelq2(m-i+1, n-i+1, a.Off(i-1, i-1), tau.Off(i-1), work); err != nil {
			panic(err)
		}
	}

	work.SetRe(0, float64(iws))

	return
}
