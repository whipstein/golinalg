package golapack

import (
	"fmt"

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
func Dlasdq(uplo mat.MatUplo, sqre, n, ncvt, nru, ncc int, d, e *mat.Vector, vt, u, c *mat.Matrix, work *mat.Vector) (info int, err error) {
	var rotate bool
	var cs, r, smin, sn, zero float64
	var i, isub, iuplo, j, np1, sqre1 int

	zero = 0.0

	//     Test the input parameters.
	iuplo = 0
	if uplo == Upper {
		iuplo = 1
	}
	if uplo == Lower {
		iuplo = 2
	}
	if iuplo == 0 {
		err = fmt.Errorf("iuplo == 0: uplo=%s", uplo)
	} else if (sqre < 0) || (sqre > 1) {
		err = fmt.Errorf("(sqre < 0) || (sqre > 1): sqre=%v", sqre)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ncvt < 0 {
		err = fmt.Errorf("ncvt < 0: ncvt=%v", ncvt)
	} else if nru < 0 {
		err = fmt.Errorf("nru < 0: nru=%v", nru)
	} else if ncc < 0 {
		err = fmt.Errorf("ncc < 0: ncc=%v", ncc)
	} else if (ncvt == 0 && vt.Rows < 1) || (ncvt > 0 && vt.Rows < max(1, n)) {
		err = fmt.Errorf("(ncvt == 0 && vt.Rows < 1) || (ncvt > 0 && vt.Rows < max(1, n)): ncvt=%v, vt.Rows=%v, n=%v", ncvt, vt.Rows, n)
	} else if u.Rows < max(1, nru) {
		err = fmt.Errorf("u.Rows < max(1, nru): u.Rows=%v, nru=%v", u.Rows, nru)
	} else if (ncc == 0 && c.Rows < 1) || (ncc > 0 && c.Rows < max(1, n)) {
		err = fmt.Errorf("(ncc == 0 && c.Rows < 1) || (ncc > 0 && c.Rows < max(1, n)): ncc=%v, c.Rows=%v, n=%v", ncc, c.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dlasdq", err)
		return
	}
	if n == 0 {
		return
	}

	//     ROTATE is true if any singular vectors desired, false otherwise
	rotate = (ncvt > 0) || (nru > 0) || (ncc > 0)
	np1 = n + 1
	sqre1 = sqre

	//     If matrix non-square upper bidiagonal, rotate to be lower
	//     bidiagonal.  The rotations are on the right.
	if (iuplo == 1) && (sqre1 == 1) {
		for i = 1; i <= n-1; i++ {
			cs, sn, r = Dlartg(d.Get(i-1), e.Get(i-1))
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i))
			d.Set(i, cs*d.Get(i))
			if rotate {
				work.Set(i-1, cs)
				work.Set(n+i-1, sn)
			}
		}
		cs, sn, r = Dlartg(d.Get(n-1), e.Get(n-1))
		d.Set(n-1, r)
		e.Set(n-1, zero)
		if rotate {
			work.Set(n-1, cs)
			work.Set(n+n-1, sn)
		}
		iuplo = 2
		sqre1 = 0

		//        Update singular vectors if desired.
		if ncvt > 0 {
			if err = Dlasr(Left, 'V', 'F', np1, ncvt, work, work.Off(np1-1), vt); err != nil {
				panic(err)
			}
		}
	}

	//     If matrix lower bidiagonal, rotate to be upper bidiagonal
	//     by applying Givens rotations on the left.
	if iuplo == 2 {
		for i = 1; i <= n-1; i++ {
			cs, sn, r = Dlartg(d.Get(i-1), e.Get(i-1))
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i))
			d.Set(i, cs*d.Get(i))
			if rotate {
				work.Set(i-1, cs)
				work.Set(n+i-1, sn)
			}
		}

		//        If matrix (N+1)-by-N lower bidiagonal, one additional
		//        rotation is needed.
		if sqre1 == 1 {
			cs, sn, r = Dlartg(d.Get(n-1), e.Get(n-1))
			d.Set(n-1, r)
			if rotate {
				work.Set(n-1, cs)
				work.Set(n+n-1, sn)
			}
		}

		//        Update singular vectors if desired.
		if nru > 0 {
			if sqre1 == 0 {
				if err = Dlasr(Right, 'V', 'F', nru, n, work, work.Off(np1-1), u); err != nil {
					panic(err)
				}
			} else {
				if err = Dlasr(Right, 'V', 'F', nru, np1, work, work.Off(np1-1), u); err != nil {
					panic(err)
				}
			}
		}
		if ncc > 0 {
			if sqre1 == 0 {
				if err = Dlasr(Left, 'V', 'F', n, ncc, work, work.Off(np1-1), c); err != nil {
					panic(err)
				}
			} else {
				if err = Dlasr(Left, 'V', 'F', np1, ncc, work, work.Off(np1-1), c); err != nil {
					panic(err)
				}
			}
		}
	}

	//     Call DBDSQR to compute the SVD of the reduced real
	//     N-by-N upper bidiagonal matrix.
	info, err = Dbdsqr(Upper, n, ncvt, nru, ncc, d, e, vt, u, c, work)

	//     Sort the singular values into ascending order (insertion sort on
	//     singular values, but only one transposition per singular vector)
	for i = 1; i <= n; i++ {
		//        Scan for smallest D(I).
		isub = i
		smin = d.Get(i - 1)
		for j = i + 1; j <= n; j++ {
			if d.Get(j-1) < smin {
				isub = j
				smin = d.Get(j - 1)
			}
		}
		if isub != i {
			//           Swap singular values and vectors.
			d.Set(isub-1, d.Get(i-1))
			d.Set(i-1, smin)
			if ncvt > 0 {
				goblas.Dswap(ncvt, vt.Vector(isub-1, 0), vt.Vector(i-1, 0))
			}
			if nru > 0 {
				goblas.Dswap(nru, u.Vector(0, isub-1, 1), u.Vector(0, i-1, 1))
			}
			if ncc > 0 {
				goblas.Dswap(ncc, c.Vector(isub-1, 0), c.Vector(i-1, 0))
			}
		}
	}

	return
}
