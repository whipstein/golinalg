package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zungrq generates an M-by-N complex matrix Q with orthonormal rows,
// which is defined as the last M rows of a product of K elementary
// reflectors of order N
//
//       Q  =  H(1)**H H(2)**H . . . H(k)**H
//
// as returned by ZGERQF.
func Zungrq(m, n, k int, a *mat.CMatrix, tau, work *mat.CVector, lwork int) (err error) {
	var lquery bool
	var zero complex128
	var i, ib, ii, iws, j, kk, l, ldwork, lwkopt, nb, nbmin, nx int

	zero = (0.0 + 0.0*1i)

	//     Test the input arguments
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < m {
		err = fmt.Errorf("n < m: m=%v, n=%v", m, n)
	} else if k < 0 || k > m {
		err = fmt.Errorf("k < 0 || k > m: m=%v, k=%v", m, k)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	}

	if err == nil {
		if m <= 0 {
			lwkopt = 1
		} else {
			nb = Ilaenv(1, "Zungrq", []byte{' '}, m, n, k, -1)
			lwkopt = m * nb
		}
		work.SetRe(0, float64(lwkopt))

		if lwork < max(1, m) && !lquery {
			err = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=%v, m=%v, lquery=%v", lwork, m, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zungrq", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m <= 0 {
		return
	}

	nbmin = 2
	nx = 0
	iws = m
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Zungrq", []byte{' '}, m, n, k, -1))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = m
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = lwork / ldwork
				nbmin = max(2, Ilaenv(2, "Zungrq", []byte{' '}, m, n, k, -1))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code after the first block.
		//        The last kk rows are handled by the block method.
		kk = min(k, ((k-nx+nb-1)/nb)*nb)

		//        Set A(1:m-kk,n-kk+1:n) to zero.
		for j = n - kk + 1; j <= n; j++ {
			for i = 1; i <= m-kk; i++ {
				a.Set(i-1, j-1, zero)
			}
		}
	} else {
		kk = 0
	}

	//     Use unblocked code for the first or only block.
	if err = Zungr2(m-kk, n-kk, k-kk, a, tau, work); err != nil {
		panic(err)
	}

	if kk > 0 {
		//        Use blocked code
		for i = k - kk + 1; i <= k; i += nb {
			ib = min(nb, k-i+1)
			ii = m - k + i
			if ii > 1 {
				//              Form the triangular factor of the block reflector
				//              H = H(i+ib-1) . . . H(i+1) H(i)
				Zlarft('B', 'R', n-k+i+ib-1, ib, a.Off(ii-1, 0), tau.Off(i-1), work.CMatrix(ldwork, opts))

				//              Apply H**H to A(1:m-k+i-1,1:n-k+i+ib-1) from the right
				Zlarfb(Right, ConjTrans, 'B', 'R', ii-1, n-k+i+ib-1, ib, a.Off(ii-1, 0), work.CMatrix(ldwork, opts), a, work.CMatrixOff(ib, ldwork, opts))
			}

			//           Apply H**H to columns 1:n-k+i+ib-1 of current block
			if err = Zungr2(ib, n-k+i+ib-1, ib, a.Off(ii-1, 0), tau.Off(i-1), work); err != nil {
				panic(err)
			}

			//           Set columns n-k+i+ib:n of current block to zero
			for l = n - k + i + ib; l <= n; l++ {
				for j = ii; j <= ii+ib-1; j++ {
					a.Set(j-1, l-1, zero)
				}
			}
		}
	}

	work.SetRe(0, float64(iws))

	return
}
