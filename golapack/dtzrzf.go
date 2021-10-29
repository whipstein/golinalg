package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtzrzf reduces the M-by-N ( M<=N ) real upper trapezoidal matrix A
// to upper triangular form by means of orthogonal transformations.
//
// The upper trapezoidal matrix A is factored as
//
//    A = ( R  0 ) * Z,
//
// where Z is an N-by-N orthogonal matrix and R is an M-by-M upper
// triangular matrix.
func Dtzrzf(m, n int, a *mat.Matrix, tau, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var zero float64
	var i, ib, iws, ki, kk, ldwork, lwkmin, lwkopt, m1, mu, nb, nbmin, nx int

	zero = 0.0

	//     Test the input arguments
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < m {
		err = fmt.Errorf("n < m: m=%v, n=%v", m, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}

	if err == nil {
		if m == 0 || m == n {
			lwkopt = 1
			lwkmin = 1
		} else {
			//           Determine the block size.
			nb = Ilaenv(1, "Dgerqf", []byte{' '}, m, n, -1, -1)
			lwkopt = m * nb
			lwkmin = max(1, m)
		}
		work.Set(0, float64(lwkopt))

		if lwork < lwkmin && !lquery {
			err = fmt.Errorf("lwork < lwkmin && !lquery: lwork=%v, lwkmin=%v, lquery=%v", lwork, lwkmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dtzrzf", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == 0 {
		return
	} else if m == n {
		for i = 1; i <= n; i++ {
			tau.Set(i-1, zero)
		}
		return
	}

	nbmin = 2
	nx = 1
	iws = m
	if nb > 1 && nb < m {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Dgerqf", []byte{' '}, m, n, -1, -1))
		if nx < m {
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

	if nb >= nbmin && nb < m && nx < m {
		//        Use blocked code initially.
		//        The last kk rows are handled by the block method.
		m1 = min(m+1, n)
		ki = ((m - nx - 1) / nb) * nb
		kk = min(m, ki+nb)

		for i = m - kk + ki + 1; i >= m-kk+1; i -= nb {
			ib = min(m-i+1, nb)

			//           Compute the TZ factorization of the current block
			//           A(i:i+ib-1,i:n)
			Dlatrz(ib, n-i+1, n-m, a.Off(i-1, i-1), tau.Off(i-1), work)
			if i > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Dlarzt('B', 'R', n-m, ib, a.Off(i-1, m1-1), tau.Off(i-1), work.Matrix(ldwork, opts))

				//              Apply H to A(1:i-1,i:n) from the right
				Dlarzb(Right, NoTrans, 'B', 'R', i-1, n-i+1, ib, n-m, a.Off(i-1, m1-1), work.Matrix(ldwork, opts), a.Off(0, i-1), work.MatrixOff(ib, ldwork, opts))
			}
		}
		mu = i + nb - 1
	} else {
		mu = m
	}

	//     Use unblocked code to factor the last or only block
	if mu > 0 {
		Dlatrz(mu, n, n-m, a, tau, work)
	}

	work.Set(0, float64(lwkopt))

	return
}
