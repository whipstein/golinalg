package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgetri computes the inverse of a matrix using the LU factorization
// computed by DGETRF.
//
// This method inverts U and then computes inv(A) by solving the system
// inv(A)*L = inv(U) for inv(A).
func Dgetri(n *int, a *mat.Matrix, lda *int, ipiv *[]int, work *mat.Matrix, lwork *int, info *int) {
	var lquery bool
	var one, zero float64
	var i, iws, j, jb, jj, jp, ldwork, lwkopt, nb, nbmin, nn int

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DGETRI"), []byte{' '}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	lwkopt = (*n) * nb
	work.Set(0, 0, float64(lwkopt))
	lquery = ((*lwork) == -1)
	if (*n) < 0 {
		(*info) = -1
	} else if (*lda) < maxint(1, *n) {
		(*info) = -3
	} else if (*lwork) < maxint(1, *n) && !lquery {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGETRI"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Form inv(U).  If INFO > 0 from DTRTRI, then U is singular,
	//     and the inverse is not computed.
	Dtrtri('U', 'N', n, a, lda, info)
	if (*info) > 0 {
		return
	}

	nbmin = 2
	ldwork = (*n)
	if nb > 1 && nb < (*n) {
		iws = maxint(ldwork*nb, 1)
		if (*lwork) < iws {
			nb = (*lwork) / ldwork
			nbmin = maxint(2, Ilaenv(func() *int { y := 2; return &y }(), []byte("DGETRI"), []byte{' '}, n, toPtr(-1), toPtr(-1), toPtr(-1)))
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
				work.SetIdx(i-1, a.Get(i-1, j-1))
				a.Set(i-1, j-1, zero)
			}

			//           Compute current column of inv(A).
			if j < (*n) {
				goblas.Dgemv(mat.NoTrans, n, toPtr((*n)-j), toPtrf64(-one), a.Off(0, j+1-1), lda, work.VectorIdx(j+1-1), toPtr(1), &one, a.Vector(0, j-1), toPtr(1))
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
					work.Set(i-1, jj-j, a.Get(i-1, jj-1))
					a.Set(i-1, jj-1, zero)
				}
			}

			//           Compute current block column of inv(A).
			if j+jb <= (*n) {
				goblas.Dgemm(mat.NoTrans, mat.NoTrans, n, &jb, toPtr((*n)-j-jb+1), toPtrf64(-one), a.Off(0, j+jb-1), lda, work.OffIdx(j+jb-1), &ldwork, &one, a.Off(0, j-1), lda)
			}
			goblas.Dtrsm(mat.Right, mat.Lower, mat.NoTrans, mat.Unit, n, &jb, &one, work.OffIdx(j-1), &ldwork, a.Off(0, j-1), lda)
		}
	}

	//     Apply column interchanges.
	for j = (*n) - 1; j >= 1; j-- {
		jp = (*ipiv)[j-1]
		if jp != j {
			goblas.Dswap(n, a.Vector(0, j-1), toPtr(1), a.Vector(0, jp-1), toPtr(1))
		}
	}

	work.Set(0, 0, float64(iws))
}
