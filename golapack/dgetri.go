package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgetri computes the inverse of a matrix using the LU factorization
// computed by DGETRF.
//
// This method inverts U and then computes inv(A) by solving the system
// inv(A)*L = inv(U) for inv(A).
func Dgetri(n int, a *mat.Matrix, ipiv []int, work *mat.Matrix) (info int, err error) {
	var lquery bool
	var one, zero float64
	var i, iws, j, jb, jj, jp, ldwork, lwkopt, nb, nbmin, nn int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	nb = Ilaenv(1, "Dgetri", []byte{' '}, n, -1, -1, -1)
	lwkopt = n * nb
	work.Set(0, 0, float64(lwkopt))
	lquery = (work.Rows == -1)
	if n < 0 {
		info = -1
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		info = -3
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if work.Rows < max(1, n) && !lquery {
		info = -6
		err = fmt.Errorf("work.Rows < max(1, n) && !lquery: work.Rows=%v, n=%v", work.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dgetri", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Form inv(U).  If INFO > 0 from DTRTRI, then U is singular,
	//     and the inverse is not computed.
	if info, err = Dtrtri(Upper, NonUnit, n, a); err != nil {
		panic(err)
	}
	if info > 0 {
		return
	}

	nbmin = 2
	ldwork = n
	if nb > 1 && nb < n {
		iws = max(ldwork*nb, 1)
		if work.Rows < iws {
			nb = work.Rows / ldwork
			nbmin = max(2, Ilaenv(2, "Dgetri", []byte{' '}, n, -1, -1, -1))
		}
	} else {
		iws = n
	}

	//     Solve the equation inv(A)*L = inv(U) for inv(A).
	if nb < nbmin || nb >= n {
		//        Use unblocked code.
		for j = n; j >= 1; j-- {
			//           Copy current column of L to WORK and replace with zeros.
			for i = j + 1; i <= n; i++ {
				work.SetIdx(i-1, a.Get(i-1, j-1))
				a.Set(i-1, j-1, zero)
			}

			//           Compute current column of inv(A).
			if j < n {
				err = a.Off(0, j-1).Vector().Gemv(NoTrans, n, n-j, -one, a.Off(0, j), work.OffIdx(j).Vector(), 1, one, 1)
			}
		}
	} else {
		//        Use blocked code.
		nn = ((n-1)/nb)*nb + 1
		for j = nn; j >= 1; j -= nb {
			jb = min(nb, n-j+1)

			//           Copy current block column of L to WORK and replace with
			//           zeros.
			for jj = j; jj <= j+jb-1; jj++ {
				for i = jj + 1; i <= n; i++ {
					work.Set(i-1, jj-j, a.Get(i-1, jj-1))
					a.Set(i-1, jj-1, zero)
				}
			}

			//           Compute current block column of inv(A).
			if j+jb <= n {
				err = a.Off(0, j-1).Gemm(NoTrans, NoTrans, n, jb, n-j-jb+1, -one, a.Off(0, j+jb-1), work.OffIdx(j+jb-1), one)
			}
			err = a.Off(0, j-1).Trsm(Right, Lower, NoTrans, Unit, n, jb, one, work.OffIdx(j-1))
		}
	}

	//     Apply column interchanges.
	for j = n - 1; j >= 1; j-- {
		jp = ipiv[j-1]
		if jp != j {
			a.Off(0, jp-1).Vector().Swap(n, a.Off(0, j-1).Vector(), 1, 1)
		}
	}

	work.Set(0, 0, float64(iws))

	return
}
