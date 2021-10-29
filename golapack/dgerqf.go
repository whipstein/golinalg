package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgerqf computes an RQ factorization of a real M-by-N matrix A:
// A = R * Q.
func Dgerqf(m, n int, a *mat.Matrix, tau, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var i, ib, iws, k, ki, kk, ldwork, lwkopt, mu, nb, nbmin, nu, nx int

	//     Test the input arguments
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}

	if err == nil {
		k = min(m, n)
		if k == 0 {
			lwkopt = 1
		} else {
			nb = Ilaenv(1, "Dgerqf", []byte{' '}, m, n, -1, -1)
			lwkopt = m * nb
		}
		work.Set(0, float64(lwkopt))
		//
		if lwork < max(1, m) && !lquery {
			err = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=%v, m=%v, lquery=%v", lwork, m, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dgerqf", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if k == 0 {
		return
	}

	nbmin = 2
	nx = 1
	iws = m
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Dgerqf", []byte{' '}, m, n, -1, -1))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = m
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = lwork / ldwork
				nbmin = max(2, Ilaenv(2, "Dgerqf", []byte{' '}, m, n, -1, -1))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code initially.
		//        The last kk rows are handled by the block method.
		ki = ((k - nx - 1) / nb) * nb
		kk = min(k, ki+nb)

		for i = k - kk + ki + 1; i >= k-kk+1; i -= nb {
			ib = min(k-i+1, nb)

			//           Compute the RQ factorization of the current block
			//           A(m-k+i:m-k+i+ib-1,1:n-k+i+ib-1)
			if err = Dgerq2(ib, n-k+i+ib-1, a.Off(m-k+i-1, 0), tau.Off(i-1), work); err != nil {
				panic(err)
			}
			if m-k+i > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Dlarft('B', 'R', n-k+i+ib-1, ib, a.Off(m-k+i-1, 0), tau.Off(i-1), work.Matrix(ldwork, opts))

				//              Apply H to A(1:m-k+i-1,1:n-k+i+ib-1) from the right
				Dlarfb(Right, NoTrans, 'B', 'R', m-k+i-1, n-k+i+ib-1, ib, a.Off(m-k+i-1, 0), work.Matrix(ldwork, opts), a, work.MatrixOff(ib, ldwork, opts))
			}
		}
		mu = m - k + i + nb - 1
		nu = n - k + i + nb - 1
	} else {
		mu = m
		nu = n
	}

	//     Use unblocked code to factor the last or only block
	if mu > 0 && nu > 0 {
		if err = Dgerq2(mu, nu, a, tau, work); err != nil {
			panic(err)
		}
	}

	work.Set(0, float64(iws))

	return
}
