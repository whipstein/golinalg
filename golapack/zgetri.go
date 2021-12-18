package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgetri computes the inverse of a matrix using the LU factorization
// computed by ZGETRF.
//
// This method inverts U and then computes inv(A) by solving the system
// inv(A)*L = inv(U) for inv(A).
func Zgetri(n int, a *mat.CMatrix, ipiv *[]int, work *mat.CVector, lwork int) (info int, err error) {
	var lquery bool
	var one, zero complex128
	var i, iws, j, jb, jj, jp, ldwork, lwkopt, nb, nbmin, nn int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	nb = Ilaenv(1, "Zgetri", []byte{' '}, n, -1, -1, -1)
	lwkopt = n * nb
	work.SetRe(0, float64(lwkopt))
	lquery = (lwork == -1)
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if lwork < max(1, n) && !lquery {
		err = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Zgetri", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Form inv(U).  If INFO > 0 from ZTRTRI, then U is singular,
	//     and the inverse is not computed.
	if info, err = Ztrtri(Upper, NonUnit, n, a); err != nil {
		panic(err)
	}
	if info > 0 {
		return
	}

	nbmin = 2
	ldwork = n
	if nb > 1 && nb < n {
		iws = max(ldwork*nb, 1)
		if lwork < iws {
			nb = lwork / ldwork
			nbmin = max(2, Ilaenv(2, "Zgetri", []byte{' '}, n, -1, -1, -1))
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
				work.Set(i-1, a.Get(i-1, j-1))
				a.Set(i-1, j-1, zero)
			}

			//           Compute current column of inv(A).
			if j < n {
				if a.Opts.Major == mat.Col {
					err = a.Off(0, j-1).CVector().Gemv(NoTrans, n, n-j, -one, a.Off(0, j), work.Off(j), 1, one, 1)
				} else {
					err = a.Off(0, j-1).CVector().Gemv(NoTrans, n, n-j, -one, a.Off(0, j), work.Off(j), 1, one, a.Cols)
				}
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
					work.Set(i+(jj-j)*ldwork-1, a.Get(i-1, jj-1))
					a.Set(i-1, jj-1, zero)
				}
			}

			//           Compute current block column of inv(A).
			if j+jb <= n {
				if err = a.Off(0, j-1).Gemm(NoTrans, NoTrans, n, jb, n-j-jb+1, -one, a.Off(0, j+jb-1), work.Off(j+jb-1).CMatrix(ldwork, opts), one); err != nil {
					panic(err)
				}
			}
			if err = a.Off(0, j-1).Trsm(Right, Lower, NoTrans, Unit, n, jb, one, work.Off(j-1).CMatrix(ldwork, opts)); err != nil {
				panic(err)
			}
		}
	}

	//     Apply column interchanges.
	for j = n - 1; j >= 1; j-- {
		jp = (*ipiv)[j-1]
		if jp != j {
			if a.Opts.Major == mat.Col {
				a.Off(0, jp-1).CVector().Swap(n, a.Off(0, j-1).CVector(), 1, 1)
			} else {
				a.Off(0, jp-1).CVector().Swap(n, a.Off(0, j-1).CVector(), a.Cols, a.Cols)
			}
		}
	}

	work.SetRe(0, float64(iws))

	return
}
