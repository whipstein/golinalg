package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhetrsaa solves a system of linear equations A*X = B with a complex
// hermitian matrix A using the factorization A = U**H*T*U or
// A = L*T*L**H computed by ZHETRF_AA.
func Zhetrsaa(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var lquery, upper bool
	var one complex128
	var k, kp, lwkopt int

	one = 1.0

	(*info) = 0
	upper = uplo == 'U'
	lquery = ((*lwork) == -1)
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	} else if (*lwork) < maxint(1, 3*(*n)-2) && !lquery {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRS_AA"), -(*info))
		return
	} else if lquery {
		lwkopt = (3*(*n) - 2)
		work.SetRe(0, float64(lwkopt))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if upper {
		//        Solve A*X = B, where A = U**H*T*U.
		//
		//        1) Forward substitution with U**H
		if (*n) > 1 {
			//           Pivot, P**T * B -> B
			for k = 1; k <= (*n); k++ {
				kp = (*ipiv)[k-1]
				if kp != k {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
			}

			//           Compute U**H \ B -> B    [ (U**H \P**T * B) ]
			goblas.Ztrsm(Left, Upper, ConjTrans, Unit, toPtr((*n)-1), nrhs, &one, a.Off(0, 1), lda, b.Off(1, 0), ldb)
		}

		//        2) Solve with triangular matrix T
		//
		//        Compute T \ B -> B   [ T \ (U**H \P**T * B) ]
		Zlacpy('F', func() *int { y := 1; return &y }(), n, a.Off(0, 0).UpdateRows((*lda)+1), toPtr((*lda)+1), work.CMatrixOff((*n)-1, 1, opts), func() *int { y := 1; return &y }())
		if (*n) > 1 {
			Zlacpy('F', func() *int { y := 1; return &y }(), toPtr((*n)-1), a.Off(0, 1).UpdateRows((*lda)+1), toPtr((*lda)+1), work.CMatrixOff(2*(*n)-1, 1, opts), func() *int { y := 1; return &y }())
			Zlacpy('F', func() *int { y := 1; return &y }(), toPtr((*n)-1), a.Off(0, 1).UpdateRows((*lda)+1), toPtr((*lda)+1), work.CMatrix(1, opts), func() *int { y := 1; return &y }())
			Zlacgv(toPtr((*n)-1), work, func() *int { y := 1; return &y }())
		}
		Zgtsv(n, nrhs, work, work.Off((*n)-1), work.Off(2*(*n)-1), b, ldb, info)

		//        3) Backward substitution with U
		if (*n) > 1 {
			//           Compute U \ B -> B   [ U \ (T \ (U**H \P**T * B) ) ]
			goblas.Ztrsm(Left, Upper, NoTrans, Unit, toPtr((*n)-1), nrhs, &one, a.Off(0, 1), lda, b.Off(1, 0), ldb)

			//           Pivot, P * B  [ P * (U**H \ (T \ (U \P**T * B) )) ]
			for k = (*n); k >= 1; k-- {
				kp = (*ipiv)[k-1]
				if kp != k {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
			}
		}

	} else {
		//        Solve A*X = B, where A = L*T*L**H.
		//
		//        1) Forward substitution with L
		if (*n) > 1 {
			//           Pivot, P**T * B -> B
			for k = 1; k <= (*n); k++ {
				kp = (*ipiv)[k-1]
				if kp != k {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
			}

			//           Compute L \ B -> B    [ (L \P**T * B) ]
			goblas.Ztrsm(Left, Lower, NoTrans, Unit, toPtr((*n)-1), nrhs, &one, a.Off(1, 0), lda, b.Off(1, 0), ldb)
		}

		//        2) Solve with triangular matrix T
		//
		//        Compute T \ B -> B   [ T \ (L \P**T * B) ]
		Zlacpy('F', func() *int { y := 1; return &y }(), n, a.Off(0, 0).UpdateRows((*lda)+1), toPtr((*lda)+1), work.CMatrixOff((*n)-1, 1, opts), func() *int { y := 1; return &y }())
		if (*n) > 1 {
			Zlacpy('F', func() *int { y := 1; return &y }(), toPtr((*n)-1), a.Off(1, 0).UpdateRows((*lda)+1), toPtr((*lda)+1), work.CMatrix(1, opts), func() *int { y := 1; return &y }())
			Zlacpy('F', func() *int { y := 1; return &y }(), toPtr((*n)-1), a.Off(1, 0).UpdateRows((*lda)+1), toPtr((*lda)+1), work.CMatrixOff(2*(*n)-1, 1, opts), func() *int { y := 1; return &y }())
			Zlacgv(toPtr((*n)-1), work.Off(2*(*n)-1), func() *int { y := 1; return &y }())
		}
		Zgtsv(n, nrhs, work, work.Off((*n)-1), work.Off(2*(*n)-1), b, ldb, info)

		//        3) Backward substitution with L**H
		if (*n) > 1 {
			//           Compute L**H \ B -> B   [ L**H \ (T \ (L \P**T * B) ) ]
			goblas.Ztrsm(Left, Lower, ConjTrans, Unit, toPtr((*n)-1), nrhs, &one, a.Off(1, 0), lda, b.Off(1, 0), ldb)

			//           Pivot, P * B  [ P * (L**H \ (T \ (L \P**T * B) )) ]
			for k = (*n); k >= 1; k-- {
				kp = (*ipiv)[k-1]
				if kp != k {
					goblas.Zswap(nrhs, b.CVector(k-1, 0), ldb, b.CVector(kp-1, 0), ldb)
				}
			}
		}

	}
}