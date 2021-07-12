package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlasdq computes the singular value decomposition (SVD) of a real
// (upper or lower) bidiagonal matrix with diagonal D and offdiagonal
// E, accumulating the transformations if desired. Letting B denote
// the input bidiagonal matrix, the algorithm computes orthogonal
// matrices Q and P such that B = Q * S * P**T (P**T denotes the transpose
// of P). The singular values S are overwritten on D.
//
// The input matrix U  is changed to U  * Q  if desired.
// The input matrix VT is changed to P**T * VT if desired.
// The input matrix C  is changed to Q**T * C  if desired.
//
// See "Computing  Small Singular Values of Bidiagonal Matrices With
// Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
// LAPACK Working Note #3, for a detailed description of the algorithm.
func Dlasdq(uplo byte, sqre, n, ncvt, nru, ncc *int, d, e *mat.Vector, vt *mat.Matrix, ldvt *int, u *mat.Matrix, ldu *int, c *mat.Matrix, ldc *int, work *mat.Vector, info *int) {
	var rotate bool
	var cs, r, smin, sn, zero float64
	var i, isub, iuplo, j, np1, sqre1 int

	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	iuplo = 0
	if uplo == 'U' {
		iuplo = 1
	}
	if uplo == 'L' {
		iuplo = 2
	}
	if iuplo == 0 {
		(*info) = -1
	} else if ((*sqre) < 0) || ((*sqre) > 1) {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ncvt) < 0 {
		(*info) = -4
	} else if (*nru) < 0 {
		(*info) = -5
	} else if (*ncc) < 0 {
		(*info) = -6
	} else if ((*ncvt) == 0 && (*ldvt) < 1) || ((*ncvt) > 0 && (*ldvt) < max(1, *n)) {
		(*info) = -10
	} else if (*ldu) < max(1, *nru) {
		(*info) = -12
	} else if ((*ncc) == 0 && (*ldc) < 1) || ((*ncc) > 0 && (*ldc) < max(1, *n)) {
		(*info) = -14
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLASDQ"), -(*info))
		return
	}
	if (*n) == 0 {
		return
	}

	//     ROTATE is true if any singular vectors desired, false otherwise
	rotate = ((*ncvt) > 0) || ((*nru) > 0) || ((*ncc) > 0)
	np1 = (*n) + 1
	sqre1 = (*sqre)

	//     If matrix non-square upper bidiagonal, rotate to be lower
	//     bidiagonal.  The rotations are on the right.
	if (iuplo == 1) && (sqre1 == 1) {
		for i = 1; i <= (*n)-1; i++ {
			Dlartg(d.GetPtr(i-1), e.GetPtr(i-1), &cs, &sn, &r)
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i))
			d.Set(i, cs*d.Get(i))
			if rotate {
				work.Set(i-1, cs)
				work.Set((*n)+i-1, sn)
			}
		}
		Dlartg(d.GetPtr((*n)-1), e.GetPtr((*n)-1), &cs, &sn, &r)
		d.Set((*n)-1, r)
		e.Set((*n)-1, zero)
		if rotate {
			work.Set((*n)-1, cs)
			work.Set((*n)+(*n)-1, sn)
		}
		iuplo = 2
		sqre1 = 0

		//        Update singular vectors if desired.
		if (*ncvt) > 0 {
			Dlasr('L', 'V', 'F', &np1, ncvt, work, work.Off(np1-1), vt, ldvt)
		}
	}

	//     If matrix lower bidiagonal, rotate to be upper bidiagonal
	//     by applying Givens rotations on the left.
	if iuplo == 2 {
		for i = 1; i <= (*n)-1; i++ {
			Dlartg(d.GetPtr(i-1), e.GetPtr(i-1), &cs, &sn, &r)
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i))
			d.Set(i, cs*d.Get(i))
			if rotate {
				work.Set(i-1, cs)
				work.Set((*n)+i-1, sn)
			}
		}

		//        If matrix (N+1)-by-N lower bidiagonal, one additional
		//        rotation is needed.
		if sqre1 == 1 {
			Dlartg(d.GetPtr((*n)-1), e.GetPtr((*n)-1), &cs, &sn, &r)
			d.Set((*n)-1, r)
			if rotate {
				work.Set((*n)-1, cs)
				work.Set((*n)+(*n)-1, sn)
			}
		}

		//        Update singular vectors if desired.
		if (*nru) > 0 {
			if sqre1 == 0 {
				Dlasr('R', 'V', 'F', nru, n, work, work.Off(np1-1), u, ldu)
			} else {
				Dlasr('R', 'V', 'F', nru, &np1, work, work.Off(np1-1), u, ldu)
			}
		}
		if (*ncc) > 0 {
			if sqre1 == 0 {
				Dlasr('L', 'V', 'F', n, ncc, work, work.Off(np1-1), c, ldc)
			} else {
				Dlasr('L', 'V', 'F', &np1, ncc, work, work.Off(np1-1), c, ldc)
			}
		}
	}

	//     Call DBDSQR to compute the SVD of the reduced real
	//     N-by-N upper bidiagonal matrix.
	Dbdsqr('U', n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info)

	//     Sort the singular values into ascending order (insertion sort on
	//     singular values, but only one transposition per singular vector)
	for i = 1; i <= (*n); i++ {
		//        Scan for smallest D(I).
		isub = i
		smin = d.Get(i - 1)
		for j = i + 1; j <= (*n); j++ {
			if d.Get(j-1) < smin {
				isub = j
				smin = d.Get(j - 1)
			}
		}
		if isub != i {
			//           Swap singular values and vectors.
			d.Set(isub-1, d.Get(i-1))
			d.Set(i-1, smin)
			if (*ncvt) > 0 {
				goblas.Dswap(*ncvt, vt.Vector(isub-1, 0), vt.Vector(i-1, 0))
			}
			if (*nru) > 0 {
				goblas.Dswap(*nru, u.Vector(0, isub-1, 1), u.Vector(0, i-1, 1))
			}
			if (*ncc) > 0 {
				goblas.Dswap(*ncc, c.Vector(isub-1, 0), c.Vector(i-1, 0))
			}
		}
	}
}
