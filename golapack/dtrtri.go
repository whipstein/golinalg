package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrtri computes the inverse of a real upper or lower triangular
// matrix A.
//
// This is the Level 3 BLAS version of the algorithm.
func Dtrtri(uplo, diag byte, n *int, a *mat.Matrix, lda *int, info *int) {
	var nounit, upper bool
	var one, zero float64
	var j, jb, nb, nn int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

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
		gltest.Xerbla([]byte("DTRTRI"), -(*info))
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
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DTRTRI"), []byte{uplo, diag}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if nb <= 1 || nb >= (*n) {
		//        Use unblocked code
		Dtrti2(uplo, diag, n, a, lda, info)
	} else {
		//        Use blocked code
		if upper {
			//           Compute inverse of upper triangular matrix
			for j = 1; j <= (*n); j += nb {
				jb = minint(nb, (*n)-j+1)

				//              Compute rows 1:j-1 of current block column
				err = goblas.Dtrmm(mat.Left, mat.Upper, mat.NoTrans, mat.DiagByte(diag), j-1, jb, one, a, *lda, a.Off(0, j-1), *lda)
				err = goblas.Dtrsm(mat.Right, mat.Upper, mat.NoTrans, mat.DiagByte(diag), j-1, jb, -one, a.Off(j-1, j-1), *lda, a.Off(0, j-1), *lda)

				//              Compute inverse of current diagonal block
				Dtrti2('U', diag, &jb, a.Off(j-1, j-1), lda, info)
			}
		} else {
			//           Compute inverse of lower triangular matrix
			nn = (((*n)-1)/nb)*nb + 1
			for j = nn; j >= 1; j -= nb {
				jb = minint(nb, (*n)-j+1)
				if j+jb <= (*n) {
					//                 Compute rows j+jb:n of current block column
					err = goblas.Dtrmm(mat.Left, mat.Lower, mat.NoTrans, mat.DiagByte(diag), (*n)-j-jb+1, jb, one, a.Off(j+jb-1, j+jb-1), *lda, a.Off(j+jb-1, j-1), *lda)
					err = goblas.Dtrsm(mat.Right, mat.Lower, mat.NoTrans, mat.DiagByte(diag), (*n)-j-jb+1, jb, -one, a.Off(j-1, j-1), *lda, a.Off(j+jb-1, j-1), *lda)
				}

				//              Compute inverse of current diagonal block
				Dtrti2('L', diag, &jb, a.Off(j-1, j-1), lda, info)
			}
		}
	}
}
