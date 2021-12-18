package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeqlf computes a QL factorization of a real M-by-N matrix A:
// A = Q * L.
func Dgeqlf(m, n int, a *mat.Matrix, tau, work *mat.Vector, lwork int) (err error) {
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
			nb = Ilaenv(1, "Dgeqlf", []byte{' '}, m, n, -1, -1)
			lwkopt = n * nb
		}
		work.Set(0, float64(lwkopt))

		if lwork < max(1, n) && !lquery {
			err = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dgeqlf", err)
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
	iws = n
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Dgeqlf", []byte{' '}, m, n, -1, -1))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = n
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = lwork / ldwork
				nbmin = max(2, Ilaenv(2, "Dgeqlf", []byte{' '}, m, n, -1, -1))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code initially.
		//        The last kk columns are handled by the block method.
		ki = ((k - nx - 1) / nb) * nb
		kk = min(k, ki+nb)

		for i = k - kk + ki + 1; i >= k-kk+1; i -= nb {
			ib = min(k-i+1, nb)

			//           Compute the QL factorization of the current block
			//           A(1:m-k+i+ib-1,n-k+i:n-k+i+ib-1)
			if err = Dgeql2(m-k+i+ib-1, ib, a.Off(0, n-k+i-1), tau.Off(i-1), work); err != nil {
				panic(err)
			}
			if n-k+i > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Dlarft('B', 'C', m-k+i+ib-1, ib, a.Off(0, n-k+i-1), tau.Off(i-1), work.Matrix(ldwork, opts))

				//              Apply H**T to A(1:m-k+i+ib-1,1:n-k+i-1) from the left
				Dlarfb(Left, Trans, 'B', 'C', m-k+i+ib-1, n-k+i-1, ib, a.Off(0, n-k+i-1), work.Matrix(ldwork, opts), a, work.Off(ib).Matrix(ldwork, opts))
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
		if err = Dgeql2(mu, nu, a, tau, work); err != nil {
			panic(err)
		}
	}

	work.Set(0, float64(iws))

	return
}
