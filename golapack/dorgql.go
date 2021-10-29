package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorgql generates an M-by-N real matrix Q with orthonormal columns,
// which is defined as the last N columns of a product of K elementary
// reflectors of order M
//
//       Q  =  H(k) . . . H(2) H(1)
//
// as returned by DGEQLF.
func Dorgql(m, n, k int, a *mat.Matrix, tau, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var zero float64
	var i, ib, iws, j, kk, l, ldwork, lwkopt, nb, nbmin, nx int

	zero = 0.0

	//     Test the input arguments
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 || n > m {
		err = fmt.Errorf("n < 0 || n > m: n=%v, m=%v", n, m)
	} else if k < 0 || k > n {
		err = fmt.Errorf("k < 0 || k > n: k=%v, n=%v", k, m)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}

	if err == nil {
		if n == 0 {
			lwkopt = 1
		} else {
			nb = Ilaenv(1, "Dorgql", []byte{' '}, m, n, k, -1)
			lwkopt = n * nb
		}
		work.Set(0, float64(lwkopt))

		if lwork < max(1, n) && !lquery {
			err = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dorgql", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n <= 0 {
		return
	}

	nbmin = 2
	nx = 0
	iws = n
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Dorgql", []byte{' '}, m, n, k, -1))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = n
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = lwork / ldwork
				nbmin = max(2, Ilaenv(2, "Dorgql", []byte{' '}, m, n, k, -1))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code after the first block.
		//        The last kk columns are handled by the block method.
		kk = min(k, ((k-nx+nb-1)/nb)*nb)

		//        Set A(m-kk+1:m,1:n-kk) to zero.
		for j = 1; j <= n-kk; j++ {
			for i = m - kk + 1; i <= m; i++ {
				a.Set(i-1, j-1, zero)
			}
		}
	} else {
		kk = 0
	}

	//     Use unblocked code for the first or only block.
	if err = Dorg2l(m-kk, n-kk, k-kk, a, tau, work); err != nil {
		panic(err)
	}

	if kk > 0 {
		//        Use blocked code
		for _, i = range genIter(k-kk+1, k, nb) {
			ib = min(nb, k-i+1)
			if n-k+i > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Dlarft('B', 'C', m-k+i+ib-1, ib, a.Off(0, n-k+i-1), tau.Off(i-1), work.Matrix(ldwork, opts))

				//              Apply H to A(1:m-k+i+ib-1,1:n-k+i-1) from the left
				Dlarfb(Left, NoTrans, 'B', 'C', m-k+i+ib-1, n-k+i-1, ib, a.Off(0, n-k+i-1), work.Matrix(ldwork, opts), a, work.MatrixOff(ib, ldwork, opts))
			}

			//           Apply H to rows 1:m-k+i+ib-1 of current block
			if err = Dorg2l(m-k+i+ib-1, ib, ib, a.Off(0, n-k+i-1), tau.Off(i-1), work); err != nil {
				panic(err)
			}

			//           Set rows m-k+i+ib:m of current block to zero
			for j = n - k + i; j <= n-k+i+ib-1; j++ {
				for l = m - k + i + ib; l <= m; l++ {
					a.Set(l-1, j-1, zero)
				}
			}
		}
	}

	work.Set(0, float64(iws))

	return
}
