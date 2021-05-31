package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgetri computes the inverse of a matrix using the LU factorization
// computed by ZGETRF.
//
// This method inverts U and then computes inv(A) by solving the system
// inv(A)*L = inv(U) for inv(A).
func Zgetri(n *int, a *mat.CMatrix, lda *int, ipiv *[]int, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var one, zero complex128
	var i, iws, j, jb, jj, jp, ldwork, lwkopt, nb, nbmin, nn int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGETRI"), []byte{' '}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	lwkopt = (*n) * nb
	work.SetRe(0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*n) < 0 {
		(*info) = -1
	} else if (*lda) < maxint(1, *n) {
		(*info) = -3
	} else if (*lwork) < maxint(1, *n) && !lquery {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGETRI"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Form inv(U).  If INFO > 0 from ZTRTRI, then U is singular,
	//     and the inverse is not computed.
	Ztrtri('U', 'N', n, a, lda, info)
	if (*info) > 0 {
		return
	}

	nbmin = 2
	ldwork = (*n)
	if nb > 1 && nb < (*n) {
		iws = maxint(ldwork*nb, 1)
		if (*lwork) < iws {
			nb = (*lwork) / ldwork
			nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("ZGETRI"), []byte{' '}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
		}
	} else {
		iws = (*n)
	}

	//     Solve the equation inv(A)*L = inv(U) for inv(A).
	if nb < nbmin || nb >= (*n) {
		//        Use unblocked code.
		for j = (*n); j >= 1; j-- {
			//           Copy current column of L to WORK and replace with zeros.
			for i = j + 1; i <= (*n); i++ {
				work.Set(i-1, a.Get(i-1, j-1))
				a.Set(i-1, j-1, zero)
			}

			//           Compute current column of inv(A).
			if j < (*n) {
				goblas.Zgemv(NoTrans, n, toPtr((*n)-j), toPtrc128(-one), a.Off(0, j+1-1), lda, work.Off(j+1-1), func() *int { y := 1; return &y }(), &one, a.CVector(0, j-1), func() *int { y := 1; return &y }())
			}
		}
	} else {
		//        Use blocked code.
		nn = (((*n)-1)/nb)*nb + 1
		for j = nn; j >= 1; j -= nb {
			jb = minint(nb, (*n)-j+1)

			//           Copy current block column of L to WORK and replace with
			//           zeros.
			for jj = j; jj <= j+jb-1; jj++ {
				for i = jj + 1; i <= (*n); i++ {
					work.Set(i+(jj-j)*ldwork-1, a.Get(i-1, jj-1))
					a.Set(i-1, jj-1, zero)
				}
			}

			//           Compute current block column of inv(A).
			if j+jb <= (*n) {
				goblas.Zgemm(NoTrans, NoTrans, n, &jb, toPtr((*n)-j-jb+1), toPtrc128(-one), a.Off(0, j+jb-1), lda, work.CMatrixOff(j+jb-1, ldwork, opts), &ldwork, &one, a.Off(0, j-1), lda)
			}
			goblas.Ztrsm(Right, Lower, NoTrans, Unit, n, &jb, &one, work.CMatrixOff(j-1, ldwork, opts), &ldwork, a.Off(0, j-1), lda)
		}
	}

	//     Apply column interchanges.
	for j = (*n) - 1; j >= 1; j-- {
		jp = (*ipiv)[j-1]
		if jp != j {
			goblas.Zswap(n, a.CVector(0, j-1), func() *int { y := 1; return &y }(), a.CVector(0, jp-1), func() *int { y := 1; return &y }())
		}
	}

	work.SetRe(0, float64(iws))
}
