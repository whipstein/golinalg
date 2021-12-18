package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorglq generates an M-by-N real matrix Q with orthonormal rows,
// which is defined as the first M rows of a product of K elementary
// reflectors of order N
//
//       Q  =  H(k) . . . H(2) H(1)
//
// as returned by DGELQF.
func Dorglq(m, n, k int, a *mat.Matrix, tau, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var zero float64
	var i, ib, iws, j, ki, kk, l, ldwork, lwkopt, nb, nbmin, nx int

	zero = 0.0

	//     Test the input arguments
	nb = Ilaenv(1, "Dorglq", []byte{' '}, m, n, k, -1)
	lwkopt = max(1, m) * nb
	work.Set(0, float64(lwkopt))
	lquery = (lwork == -1)
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < m {
		err = fmt.Errorf("n < m: n=%v, m=%v", n, m)
	} else if k < 0 || k > m {
		err = fmt.Errorf("k < 0 || k > m: k=%v, m=%v", k, m)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if lwork < max(1, m) && !lquery {
		err = fmt.Errorf("lwork < max(1, m) && !lquery: lwork=%v, m=%v, lquery=%v", lwork, m, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Dorglq", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m <= 0 {
		work.Set(0, 1)
		return
	}

	nbmin = 2
	nx = 0
	iws = m
	if nb > 1 && nb < k {
		//        Determine when to cross over from blocked to unblocked code.
		nx = max(0, Ilaenv(3, "Dorglq", []byte{' '}, m, n, k, -1))
		if nx < k {
			//           Determine if workspace is large enough for blocked code.
			ldwork = m
			iws = ldwork * nb
			if lwork < iws {
				//              Not enough workspace to use optimal NB:  reduce NB and
				//              determine the minimum value of NB.
				nb = lwork / ldwork
				nbmin = max(2, Ilaenv(2, "Dorglq", []byte{' '}, m, n, k, -1))
			}
		}
	}

	if nb >= nbmin && nb < k && nx < k {
		//        Use blocked code after the last block.
		//        The first kk rows are handled by the block method.
		ki = ((k - nx - 1) / nb) * nb
		kk = min(k, ki+nb)

		//        Set A(kk+1:m,1:kk) to zero.
		for j = 1; j <= kk; j++ {
			for i = kk + 1; i <= m; i++ {
				a.Set(i-1, j-1, zero)
			}
		}
	} else {
		kk = 0
	}

	//     Use unblocked code for the last or only block.
	if kk < m {
		if err = Dorgl2(m-kk, n-kk, k-kk, a.Off(kk, kk), tau.Off(kk), work); err != nil {
			panic(err)
		}
	}

	if kk > 0 {
		//        Use blocked code
		for i = ki + 1; i >= 1; i -= nb {
			ib = min(nb, k-i+1)
			if i+ib <= m {
				//              Form the triangular factor of the block reflector
				//              H = H(i) H(i+1) . . . H(i+ib-1)
				Dlarft('F', 'R', n-i+1, ib, a.Off(i-1, i-1), tau.Off(i-1), work.Matrix(ldwork, opts))

				//              Apply H**T to A(i+ib:m,i:n) from the right
				Dlarfb(Right, Trans, 'F', 'R', m-i-ib+1, n-i+1, ib, a.Off(i-1, i-1), work.Matrix(ldwork, opts), a.Off(i+ib-1, i-1), work.Off(ib).Matrix(ldwork, opts))
			}

			//           Apply H**T to columns i:n of current block
			if err = Dorgl2(ib, n-i+1, ib, a.Off(i-1, i-1), tau.Off(i-1), work); err != nil {
				panic(err)
			}

			//           Set columns 1:i-1 of current block to zero
			for j = 1; j <= i-1; j++ {
				for l = i; l <= i+ib-1; l++ {
					a.Set(l-1, j-1, zero)
				}
			}
		}
	}

	work.Set(0, float64(iws))

	return
}
