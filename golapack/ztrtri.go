package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Ztrtri computes the inverse of a complex upper or lower triangular
// matrix A.
//
// This is the Level 3 BLAS version of the algorithm.
func Ztrtri(uplo, diag byte, n *int, a *mat.CMatrix, lda, info *int) {
	var nounit, upper bool
	var one, zero complex128
	var j, jb, nb, nn int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	nounit = diag == 'N'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if !nounit && diag != 'U' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTRTRI"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Check for singularity if non-unit.
	if nounit {
		for (*info) = 1; (*info) <= (*n); (*info)++ {
			if a.Get((*info)-1, (*info)-1) == zero {
				return
			}
		}
		(*info) = 0
	}

	//     Determine the block size for this environment.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("ZTRTRI"), []byte{uplo, diag}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if nb <= 1 || nb >= (*n) {
		//        Use unblocked code
		Ztrti2(uplo, diag, n, a, lda, info)
	} else {
		//        Use blocked code
		if upper {
			//           Compute inverse of upper triangular matrix
			for j = 1; j <= (*n); j += nb {
				jb = minint(nb, (*n)-j+1)

				//              Compute rows 1:j-1 of current block column
				goblas.Ztrmm(Left, Upper, NoTrans, mat.DiagByte(diag), toPtr(j-1), &jb, &one, a, lda, a.Off(0, j-1), lda)
				goblas.Ztrsm(Right, Upper, NoTrans, mat.DiagByte(diag), toPtr(j-1), &jb, toPtrc128(-one), a.Off(j-1, j-1), lda, a.Off(0, j-1), lda)

				//              Compute inverse of current diagonal block
				Ztrti2('U', diag, &jb, a.Off(j-1, j-1), lda, info)
			}
		} else {
			//           Compute inverse of lower triangular matrix
			nn = (((*n)-1)/nb)*nb + 1
			for j = nn; j >= 1; j -= nb {
				jb = minint(nb, (*n)-j+1)
				if j+jb <= (*n) {
					//                 Compute rows j+jb:n of current block column
					goblas.Ztrmm(Left, Lower, NoTrans, mat.DiagByte(diag), toPtr((*n)-j-jb+1), &jb, &one, a.Off(j+jb-1, j+jb-1), lda, a.Off(j+jb-1, j-1), lda)
					goblas.Ztrsm(Right, Lower, NoTrans, mat.DiagByte(diag), toPtr((*n)-j-jb+1), &jb, toPtrc128(-one), a.Off(j-1, j-1), lda, a.Off(j+jb-1, j-1), lda)
				}

				//              Compute inverse of current diagonal block
				Ztrti2('L', diag, &jb, a.Off(j-1, j-1), lda, info)
			}
		}
	}
}
