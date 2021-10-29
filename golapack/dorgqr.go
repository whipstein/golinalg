package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorgqr generates an M-by-N real matrix Q with orthonormal columns,
// which is defined as the first N columns of a product of K elementary
// reflectors of order M
//
//       Q  =  H(1) H(2) . . . H(k)
//
// as returned by DGEQRF.
func Dorgqr(m, n, k int, a *mat.Matrix, tau, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var zero float64
	var i, ib, iws, j, ki, kk, l, ldwork, lwkopt, nb, nbmin, nx int

	zero = 0.0

	//     Test the input arguments
	nb = Ilaenv(1, "Dorgqr", []byte{' '}, m, n, k, -1)
	lwkopt = max(1, n) * nb
	work.Set(0, float64(lwkopt))
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || n > m {
		err = fmt.Errorf("n < 0 || n > m: n=%v, m=%v", n, m)
	} else if k < 0 || k > n {
		err = fmt.Errorf("k < 0 || k > n: k=%v, n=%v", k, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if lwork < max(1, n) && !lquery {
		err = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Dorgqr", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n <= 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	nx = 0
	iws = n
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Dorgqr", []byte{' '}, m, n, k, -1))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = n
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = lwork / ldwork
				nbmin = max(2, Ilaenv(2, "Dorgqr", []byte{' '}, m, n, k, -1))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code after the last block.
		//        The first kk columns are handled by the block method.
		ki = ((k - nx - 1) / nb) * nb
		kk = min(k, ki+nb)

		//        Set A(1:kk,kk+1:n) to zero.
		for j = kk + 1; j <= n; j++ {
			for i = 1; i <= kk; i++ {
				a.Set(i-1, j-1, zero)
			}
		}
	} else {
		kk = 0
	}

	//     Use unblocked code for the last or only block.
	if kk < n {
		if err = Dorg2r(m-kk, n-kk, k-kk, a.Off(kk, kk), tau.Off(kk), work); err != nil {
			panic(err)
		}
	}

	if kk > 0 {
		//        Use blocked code
		for i = ki + 1; i >= 1; i -= nb {
			ib = min(nb, k-i+1)
			if i+ib <= n {
				//              Form the triangular factor of the block reflector
				//              H = H(i) H(i+1) . . . H(i+ib-1)
				Dlarft('F', 'C', m-i+1, ib, a.Off(i-1, i-1), tau.Off(i-1), work.Matrix(ldwork, opts))

				//              Apply H to A(i:m,i+ib:n) from the left
				Dlarfb(Left, NoTrans, 'F', 'C', m-i+1, n-i-ib+1, ib, a.Off(i-1, i-1), work.Matrix(ldwork, opts), a.Off(i-1, i+ib-1), work.MatrixOff(ib, ldwork, opts))
			}

			//           Apply H to rows i:m of current block
			if err = Dorg2r(m-i+1, ib, ib, a.Off(i-1, i-1), tau.Off(i-1), work); err != nil {
				panic(err)
			}

			//           Set rows 1:i-1 of current block to zero
			for j = i; j <= i+ib-1; j++ {
				for l = 1; l <= i-1; l++ {
					a.Set(l-1, j-1, zero)
				}
			}
		}
	}

	work.Set(0, float64(iws))

	return
}
