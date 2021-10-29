package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsteqr computes all eigenvalues and, optionally, eigenvectors of a
// symmetric tridiagonal matrix using the implicit QL or QR method.
// The eigenvectors of a full or band complex Hermitian matrix can also
// be found if ZHETRD or ZHPTRD or ZHBTRD has been used to reduce this
// matrix to tridiagonal form.
func Zsteqr(compz byte, n int, d, e *mat.Vector, z *mat.CMatrix, work *mat.Vector) (info int, err error) {
	var cone, czero complex128
	var anorm, b, c, eps, eps2, f, g, one, p, r, rt1, rt2, s, safmax, safmin, ssfmax, ssfmin, three, tst, two, zero float64
	var i, icompz, ii, iscale, j, jtot, k, l, l1, lend, lendm1, lendp1, lendsv, lm1, lsv, m, maxit, mm, mm1, nm1, nmaxit int

	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	maxit = 30

	//     Test the input parameters.
	if compz == 'N' {
		icompz = 0
	} else if compz == 'V' {
		icompz = 1
	} else if compz == 'I' {
		icompz = 2
	} else {
		icompz = -1
	}
	if icompz < 0 {
		err = fmt.Errorf("icompz < 0: compz='%c'", compz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if (z.Rows < 1) || (icompz > 0 && z.Rows < max(1, n)) {
		err = fmt.Errorf("(z.Rows < 1) || (icompz > 0 && z.Rows < max(1, n)): compz='%c', z.Rows=%v, n=%v", compz, z.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zsteqr", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		if icompz == 2 {
			z.Set(0, 0, cone)
		}
		return
	}

	//     Determine the unit roundoff and over/underflow thresholds.
	eps = Dlamch(Epsilon)
	eps2 = math.Pow(eps, 2)
	safmin = Dlamch(SafeMinimum)
	safmax = one / safmin
	ssfmax = math.Sqrt(safmax) / three
	ssfmin = math.Sqrt(safmin) / eps2

	//     Compute the eigenvalues and eigenvectors of the tridiagonal
	//     matrix.
	if icompz == 2 {
		Zlaset(Full, n, n, czero, cone, z)
	}

	nmaxit = n * maxit
	jtot = 0

	//     Determine where the matrix splits and choose QL or QR iteration
	//     for each block, according to whether top or bottom diagonal
	//     element is smaller.
	l1 = 1
	nm1 = n - 1

label10:
	;
	if l1 > n {
		goto label160
	}
	if l1 > 1 {
		e.Set(l1-1-1, zero)
	}
	if l1 <= nm1 {
		for m = l1; m <= nm1; m++ {
			tst = e.GetMag(m - 1)
			if tst == zero {
				goto label30
			}
			if tst <= (math.Sqrt(d.GetMag(m-1))*math.Sqrt(d.GetMag(m)))*eps {
				e.Set(m-1, zero)
				goto label30
			}
		}
	}
	m = n

label30:
	;
	l = l1
	lsv = l
	lend = m
	lendsv = lend
	l1 = m + 1
	if lend == l {
		goto label10
	}

	//     Scale submatrix in rows and columns L to LEND
	anorm = Dlanst('I', lend-l+1, d.Off(l-1), e.Off(l-1))
	iscale = 0
	if anorm == zero {
		goto label10
	}
	if anorm > ssfmax {
		iscale = 1
		if err = Dlascl('G', 0, 0, anorm, ssfmax, lend-l+1, 1, d.MatrixOff(l-1, n, opts)); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, anorm, ssfmax, lend-l, 1, e.MatrixOff(l-1, n, opts)); err != nil {
			panic(err)
		}
	} else if anorm < ssfmin {
		iscale = 2
		if err = Dlascl('G', 0, 0, anorm, ssfmin, lend-l+1, 1, d.MatrixOff(l-1, n, opts)); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, anorm, ssfmin, lend-l, 1, e.MatrixOff(l-1, n, opts)); err != nil {
			panic(err)
		}
	}

	//     Choose between QL and QR iteration
	if d.GetMag(lend-1) < d.GetMag(l-1) {
		lend = lsv
		l = lendsv
	}

	if lend > l {
		//        QL Iteration
		//
		//        Look for small subdiagonal element.
	label40:
		;
		if l != lend {
			lendm1 = lend - 1
			for m = l; m <= lendm1; m++ {
				tst = math.Pow(e.GetMag(m-1), 2)
				if tst <= (eps2*d.GetMag(m-1))*d.GetMag(m)+safmin {
					goto label60
				}
			}
		}

		m = lend

	label60:
		;
		if m < lend {
			e.Set(m-1, zero)
		}
		p = d.Get(l - 1)
		if m == l {
			goto label80
		}

		//        If remaining matrix is 2-by-2, use DLAE2 or SLAEV2
		//        to compute its eigensystem.
		if m == l+1 {
			if icompz > 0 {
				rt1, rt2, c, s = Dlaev2(d.Get(l-1), e.Get(l-1), d.Get(l))
				work.Set(l-1, c)
				work.Set(n-1+l-1, s)
				if err = Zlasr(Right, 'V', 'B', n, 2, work.Off(l-1), work.Off(n-1+l-1), z.Off(0, l-1)); err != nil {
					panic(err)
				}
			} else {
				rt1, rt2 = Dlae2(d.Get(l-1), e.Get(l-1), d.Get(l))
			}
			d.Set(l-1, rt1)
			d.Set(l, rt2)
			e.Set(l-1, zero)
			l = l + 2
			if l <= lend {
				goto label40
			}
			goto label140
		}

		if jtot == nmaxit {
			goto label140
		}
		jtot = jtot + 1

		//        Form shift.
		g = (d.Get(l) - p) / (two * e.Get(l-1))
		r = Dlapy2(g, one)
		g = d.Get(m-1) - p + (e.Get(l-1) / (g + math.Copysign(r, g)))

		s = one
		c = one
		p = zero

		//        Inner loop
		mm1 = m - 1
		for i = mm1; i >= l; i -= 1 {
			f = s * e.Get(i-1)
			b = c * e.Get(i-1)
			c, s, r = Dlartg(g, f)
			if i != m-1 {
				e.Set(i, r)
			}
			g = d.Get(i) - p
			r = (d.Get(i-1)-g)*s + two*c*b
			p = s * r
			d.Set(i, g+p)
			g = c*r - b

			//           If eigenvectors are desired, then save rotations.
			if icompz > 0 {
				work.Set(i-1, c)
				work.Set(n-1+i-1, -s)
			}

		}

		//        If eigenvectors are desired, then apply saved rotations.
		if icompz > 0 {
			mm = m - l + 1
			if err = Zlasr(Right, 'V', 'B', n, mm, work.Off(l-1), work.Off(n-1+l-1), z.Off(0, l-1)); err != nil {
				panic(err)
			}
		}

		d.Set(l-1, d.Get(l-1)-p)
		e.Set(l-1, g)
		goto label40

		//        Eigenvalue found.
	label80:
		;
		d.Set(l-1, p)

		l = l + 1
		if l <= lend {
			goto label40
		}
		goto label140

	} else {
		//        QR Iteration
		//
		//        Look for small superdiagonal element.
	label90:
		;
		if l != lend {
			lendp1 = lend + 1
			for m = l; m >= lendp1; m -= 1 {
				tst = math.Pow(e.GetMag(m-1-1), 2)
				if tst <= (eps2*d.GetMag(m-1))*d.GetMag(m-1-1)+safmin {
					goto label110
				}
			}
		}

		m = lend

	label110:
		;
		if m > lend {
			e.Set(m-1-1, zero)
		}
		p = d.Get(l - 1)
		if m == l {
			goto label130
		}

		//        If remaining matrix is 2-by-2, use DLAE2 or SLAEV2
		//        to compute its eigensystem.
		if m == l-1 {
			if icompz > 0 {
				rt1, rt2, c, s = Dlaev2(d.Get(l-1-1), e.Get(l-1-1), d.Get(l-1))
				work.Set(m-1, c)
				work.Set(n-1+m-1, s)
				if err = Zlasr(Right, 'V', 'F', n, 2, work.Off(m-1), work.Off(n-1+m-1), z.Off(0, l-1-1)); err != nil {
					panic(err)
				}
			} else {
				rt1, rt2 = Dlae2(d.Get(l-1-1), e.Get(l-1-1), d.Get(l-1))
			}
			d.Set(l-1-1, rt1)
			d.Set(l-1, rt2)
			e.Set(l-1-1, zero)
			l = l - 2
			if l >= lend {
				goto label90
			}
			goto label140
		}

		if jtot == nmaxit {
			goto label140
		}
		jtot = jtot + 1

		//        Form shift.
		g = (d.Get(l-1-1) - p) / (two * e.Get(l-1-1))
		r = Dlapy2(g, one)
		g = d.Get(m-1) - p + (e.Get(l-1-1) / (g + math.Copysign(r, g)))

		s = one
		c = one
		p = zero

		//        Inner loop
		lm1 = l - 1
		for i = m; i <= lm1; i++ {
			f = s * e.Get(i-1)
			b = c * e.Get(i-1)
			c, s, r = Dlartg(g, f)
			if i != m {
				e.Set(i-1-1, r)
			}
			g = d.Get(i-1) - p
			r = (d.Get(i)-g)*s + two*c*b
			p = s * r
			d.Set(i-1, g+p)
			g = c*r - b

			//           If eigenvectors are desired, then save rotations.
			if icompz > 0 {
				work.Set(i-1, c)
				work.Set(n-1+i-1, s)
			}

		}

		//        If eigenvectors are desired, then apply saved rotations.
		if icompz > 0 {
			mm = l - m + 1
			if err = Zlasr(Right, 'V', 'F', n, mm, work.Off(m-1), work.Off(n-1+m-1), z.Off(0, m-1)); err != nil {
				panic(err)
			}
		}

		d.Set(l-1, d.Get(l-1)-p)
		e.Set(lm1-1, g)
		goto label90

		//        Eigenvalue found.
	label130:
		;
		d.Set(l-1, p)

		l = l - 1
		if l >= lend {
			goto label90
		}
		goto label140

	}

	//     Undo scaling if necessary
label140:
	;
	if iscale == 1 {
		if err = Dlascl('G', 0, 0, ssfmax, anorm, lendsv-lsv+1, 1, d.MatrixOff(lsv-1, n, opts)); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, ssfmax, anorm, lendsv-lsv, 1, e.MatrixOff(lsv-1, n, opts)); err != nil {
			panic(err)
		}
	} else if iscale == 2 {
		if err = Dlascl('G', 0, 0, ssfmin, anorm, lendsv-lsv+1, 1, d.MatrixOff(lsv-1, n, opts)); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, ssfmin, anorm, lendsv-lsv, 1, e.MatrixOff(lsv-1, n, opts)); err != nil {
			panic(err)
		}
	}

	//     Check for no convergence to an eigenvalue after a total
	//     of N*MAXIT iterations.
	if jtot == nmaxit {
		for i = 1; i <= n-1; i++ {
			if e.Get(i-1) != zero {
				info = info + 1
			}
		}
		return
	}
	goto label10

	//     Order eigenvalues and eigenvectors.
label160:
	;
	if icompz == 0 {
		//        Use Quick Sort
		if err = Dlasrt('I', n, d); err != nil {
			panic(err)
		}

	} else {
		//        Use Selection Sort to minimize swaps of eigenvectors
		for ii = 2; ii <= n; ii++ {
			i = ii - 1
			k = i
			p = d.Get(i - 1)
			for j = ii; j <= n; j++ {
				if d.Get(j-1) < p {
					k = j
					p = d.Get(j - 1)
				}
			}
			if k != i {
				d.Set(k-1, d.Get(i-1))
				d.Set(i-1, p)
				goblas.Zswap(n, z.CVector(0, i-1, 1), z.CVector(0, k-1, 1))
			}
		}
	}

	return
}
