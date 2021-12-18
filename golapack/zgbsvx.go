package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbsvx uses the LU factorization to compute the solution to a complex
// system of linear equations A * X = B, A**T * X = B, or A**H * X = B,
// where A is a band matrix of order N with KL subdiagonals and KU
// superdiagonals, and X and B are N-by-NRHS matrices.
//
// Error bounds on the solution and a condition estimate are also
// provided.
func Zgbsvx(fact byte, trans mat.MatTrans, n, kl, ku, nrhs int, ab, afb *mat.CMatrix, ipiv *[]int, equed byte, r, c *mat.Vector, b, x *mat.CMatrix, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector) (equedOut byte, rcond float64, info int, err error) {
	var colequ, equil, nofact, notran, rowequ bool
	var norm byte
	var amax, anorm, bignum, colcnd, one, rcmax, rcmin, rowcnd, rpvgrw, smlnum, zero float64
	var i, infequ, j, j1, j2 int

	zero = 0.0
	one = 1.0
	equedOut = equed

	nofact = fact == 'N'
	equil = fact == 'E'
	notran = trans == NoTrans
	if nofact || equil {
		equedOut = 'N'
		rowequ = false
		colequ = false
	} else {
		rowequ = equedOut == 'R' || equedOut == 'B'
		colequ = equedOut == 'C' || equedOut == 'B'
		smlnum = Dlamch(SafeMinimum)
		bignum = one / smlnum
	}

	//     Test the input parameters.
	if !nofact && !equil && fact != 'F' {
		err = fmt.Errorf("!nofact && !equil && fact != 'F': fact='%c'", fact)
	} else if !notran && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kl < 0 {
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku < 0: ku=%v", ku)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if ab.Rows < kl+ku+1 {
		err = fmt.Errorf("ab.Rows < kl+ku+1: ab.Rows=%v, kl=%v, ku=%v", ab.Rows, kl, ku)
	} else if afb.Rows < 2*kl+ku+1 {
		err = fmt.Errorf("afb.Rows < 2*kl+ku+1: afb.Rows=%v, kl=%v, ku=%v", afb.Rows, kl, ku)
	} else if fact == 'F' && !(rowequ || colequ || equedOut == 'N') {
		err = fmt.Errorf("fact == 'F' && !(rowequ || colequ || equed == 'N'): fact='%c', equed='%c'", fact, equedOut)
	} else {
		if rowequ {
			rcmin = bignum
			rcmax = zero
			for j = 1; j <= n; j++ {
				rcmin = math.Min(rcmin, r.Get(j-1))
				rcmax = math.Max(rcmax, r.Get(j-1))
			}
			if rcmin <= zero {
				err = fmt.Errorf("rcmin <= zero: rcmin=%v", rcmin)
			} else if n > 0 {
				rowcnd = math.Max(rcmin, smlnum) / math.Min(rcmax, bignum)
			} else {
				rowcnd = one
			}
		}
		if colequ && err == nil {
			rcmin = bignum
			rcmax = zero
			for j = 1; j <= n; j++ {
				rcmin = math.Min(rcmin, c.Get(j-1))
				rcmax = math.Max(rcmax, c.Get(j-1))
			}
			if rcmin <= zero {
				err = fmt.Errorf("rcmin <= zero: rcmin=%v", rcmin)
			} else if n > 0 {
				colcnd = math.Max(rcmin, smlnum) / math.Min(rcmax, bignum)
			} else {
				colcnd = one
			}
		}
		if err == nil {
			if b.Rows < max(1, n) {
				err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
			} else if x.Rows < max(1, n) {
				err = fmt.Errorf("x.Rows < max(1, n): x.Rows=%v, n=%v", x.Rows, n)
			}
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgbsvx", err)
		return
	}

	if equil {
		//        Compute row and column scalings to equilibrate the matrix A.
		rowcnd, colcnd, amax, infequ, err = Zgbequ(n, n, kl, ku, ab, r, c)
		if infequ == 0 {
			//           Equilibrate the matrix.
			equedOut = Zlaqgb(n, n, kl, ku, ab, r, c, rowcnd, colcnd, amax)
			rowequ = equedOut == 'R' || equedOut == 'B'
			colequ = equedOut == 'C' || equedOut == 'B'
		}
	}

	//     Scale the right hand side.
	if notran {
		if rowequ {
			for j = 1; j <= nrhs; j++ {
				for i = 1; i <= n; i++ {
					b.Set(i-1, j-1, r.GetCmplx(i-1)*b.Get(i-1, j-1))
				}
			}
		}
	} else if colequ {
		for j = 1; j <= nrhs; j++ {
			for i = 1; i <= n; i++ {
				b.Set(i-1, j-1, c.GetCmplx(i-1)*b.Get(i-1, j-1))
			}
		}
	}

	if nofact || equil {
		//        Compute the LU factorization of the band matrix A.
		for j = 1; j <= n; j++ {
			j1 = max(j-ku, 1)
			j2 = min(j+kl, n)
			afb.Off(kl+ku+1-j+j1-1, j-1).CVector().Copy(j2-j1+1, ab.Off(ku+1-j+j1-1, j-1).CVector(), 1, 1)
		}

		if info, err = Zgbtrf(n, n, kl, ku, afb, ipiv); err != nil {
			panic(err)
		}

		//        Return if INFO is non-zero.
		if info > 0 {
			//           Compute the reciprocal pivot growth factor of the
			//           leading rank-deficient INFO columns of A.
			anorm = zero
			for j = 1; j <= info; j++ {
				for i = max(ku+2-j, 1); i <= min(n+ku+1-j, kl+ku+1); i++ {
					anorm = math.Max(anorm, ab.GetMag(i-1, j-1))
				}
			}
			rpvgrw = Zlantb('M', Upper, NonUnit, info, min(info-1, kl+ku), afb.Off(max(1, kl+ku+2-info)-1, 0), rwork)
			if rpvgrw == zero {
				rpvgrw = one
			} else {
				rpvgrw = anorm / rpvgrw
			}
			rwork.Set(0, rpvgrw)
			rcond = zero
			return
		}
	}

	//     Compute the norm of the matrix A and the
	//     reciprocal pivot growth factor RPVGRW.
	if notran {
		norm = '1'
	} else {
		norm = 'I'
	}
	anorm = Zlangb(norm, n, kl, ku, ab, rwork)
	rpvgrw = Zlantb('M', Upper, NonUnit, n, kl+ku, afb, rwork)
	if rpvgrw == zero {
		rpvgrw = one
	} else {
		rpvgrw = Zlangb('M', n, kl, ku, ab, rwork) / rpvgrw
	}

	//     Compute the reciprocal of the condition number of A.
	if rcond, err = Zgbcon(norm, n, kl, ku, afb, ipiv, anorm, work, rwork); err != nil {
		panic(err)
	}

	//     Compute the solution matrix X.
	Zlacpy(Full, n, nrhs, b, x)
	if err = Zgbtrs(trans, n, kl, ku, nrhs, afb, ipiv, x); err != nil {
		panic(err)
	}

	//     Use iterative refinement to improve the computed solution and
	//     compute error bounds and backward error estimates for it.
	if err = Zgbrfs(trans, n, kl, ku, nrhs, ab, afb, ipiv, b, x, ferr, berr, work, rwork); err != nil {
		panic(err)
	}

	//     Transform the solution matrix X to a solution of the original
	//     system.
	if notran {
		if colequ {
			for j = 1; j <= nrhs; j++ {
				for i = 1; i <= n; i++ {
					x.Set(i-1, j-1, c.GetCmplx(i-1)*x.Get(i-1, j-1))
				}
			}
			for j = 1; j <= nrhs; j++ {
				ferr.Set(j-1, ferr.Get(j-1)/colcnd)
			}
		}
	} else if rowequ {
		for j = 1; j <= nrhs; j++ {
			for i = 1; i <= n; i++ {
				x.Set(i-1, j-1, r.GetCmplx(i-1)*x.Get(i-1, j-1))
			}
		}
		for j = 1; j <= nrhs; j++ {
			ferr.Set(j-1, ferr.Get(j-1)/rowcnd)
		}
	}

	//     Set INFO = N+1 if the matrix is singular to working precision.
	if rcond < Dlamch(Epsilon) {
		info = n + 1
	}

	rwork.Set(0, rpvgrw)

	return
}
