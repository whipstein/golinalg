package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsteqr computes all eigenvalues and, optionally, eigenvectors of a
// symmetric tridiagonal matrix using the implicit QL or QR method.
// The eigenvectors of a full or band symmetric matrix can also be found
// if DSYTRD or DSPTRD or DSBTRD has been used to reduce this matrix to
// tridiagonal form.
func Dsteqr(compz byte, n *int, d, e *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, info *int) {
	var anorm, b, c, eps, eps2, f, g, one, p, r, rt1, rt2, s, safmax, safmin, ssfmax, ssfmin, three, tst, two, zero float64
	var i, icompz, ii, iscale, j, jtot, k, l, l1, lend, lendm1, lendp1, lendsv, lm1, lsv, m, maxit, mm, mm1, nm1, nmaxit int

	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0
	maxit = 30

	//     Test the input parameters.
	(*info) = 0

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
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if ((*ldz) < 1) || (icompz > 0 && (*ldz) < maxint(1, *n)) {
		(*info) = -6
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSTEQR"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		if icompz == 2 {
			z.Set(0, 0, one)
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
		Dlaset('F', n, n, &zero, &one, z, ldz)
	}

	nmaxit = (*n) * maxit
	jtot = 0

	//     Determine where the matrix splits and choose QL or QR iteration
	//     for each block, according to whether top or bottom diagonal
	//     element is smaller.
	l1 = 1
	nm1 = (*n) - 1

label10:
	;
	if l1 > (*n) {
		goto label160
	}
	if l1 > 1 {
		e.Set(l1-1-1, zero)
	}
	if l1 <= nm1 {
		for m = l1; m <= nm1; m++ {
			tst = math.Abs(e.Get(m - 1))
			if tst == zero {
				goto label30
			}
			if tst <= (math.Sqrt(math.Abs(d.Get(m-1)))*math.Sqrt(math.Abs(d.Get(m+1-1))))*eps {
				e.Set(m-1, zero)
				goto label30
			}
		}
	}
	m = (*n)

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
	anorm = Dlanst('M', toPtr(lend-l+1), d.Off(l-1), e.Off(l-1))
	iscale = 0
	if anorm == zero {
		goto label10
	}
	if anorm > ssfmax {
		iscale = 1
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anorm, &ssfmax, toPtr(lend-l+1), func() *int { y := 1; return &y }(), d.MatrixOff(l-1, *n, opts), n, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anorm, &ssfmax, toPtr(lend-l), func() *int { y := 1; return &y }(), e.MatrixOff(l-1, *n, opts), n, info)
	} else if anorm < ssfmin {
		iscale = 2
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anorm, &ssfmin, toPtr(lend-l+1), func() *int { y := 1; return &y }(), d.MatrixOff(l-1, *n, opts), n, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &anorm, &ssfmin, toPtr(lend-l), func() *int { y := 1; return &y }(), e.MatrixOff(l-1, *n, opts), n, info)
	}

	//     Choose between QL and QR iteration
	if math.Abs(d.Get(lend-1)) < math.Abs(d.Get(l-1)) {
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
				tst = math.Pow(math.Abs(e.Get(m-1)), 2)
				if tst <= (eps2*math.Abs(d.Get(m-1)))*math.Abs(d.Get(m+1-1))+safmin {
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
				Dlaev2(d.GetPtr(l-1), e.GetPtr(l-1), d.GetPtr(l+1-1), &rt1, &rt2, &c, &s)
				work.Set(l-1, c)
				work.Set((*n)-1+l-1, s)
				Dlasr('R', 'V', 'B', n, func() *int { y := 2; return &y }(), work.Off(l-1), work.Off((*n)-1+l-1), z.Off(0, l-1), ldz)
			} else {
				Dlae2(d.GetPtr(l-1), e.GetPtr(l-1), d.GetPtr(l+1-1), &rt1, &rt2)
			}
			d.Set(l-1, rt1)
			d.Set(l+1-1, rt2)
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
		g = (d.Get(l+1-1) - p) / (two * e.Get(l-1))
		r = Dlapy2(&g, &one)
		g = d.Get(m-1) - p + (e.Get(l-1) / (g + signf64(r, g)))

		s = one
		c = one
		p = zero

		//        Inner loop
		mm1 = m - 1
		for i = mm1; i >= l; i-- {
			f = s * e.Get(i-1)
			b = c * e.Get(i-1)
			Dlartg(&g, &f, &c, &s, &r)
			if i != m-1 {
				e.Set(i+1-1, r)
			}
			g = d.Get(i+1-1) - p
			r = (d.Get(i-1)-g)*s + two*c*b
			p = s * r
			d.Set(i+1-1, g+p)
			g = c*r - b

			//           If eigenvectors are desired, then save rotations.
			if icompz > 0 {
				work.Set(i-1, c)
				work.Set((*n)-1+i-1, -s)
			}

		}

		//        If eigenvectors are desired, then apply saved rotations.
		if icompz > 0 {
			mm = m - l + 1
			Dlasr('R', 'V', 'B', n, &mm, work.Off(l-1), work.Off((*n)-1+l-1), z.Off(0, l-1), ldz)
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
			for m = l; m >= lendp1; m-- {
				tst = math.Pow(math.Abs(e.Get(m-1-1)), 2)
				if tst <= (eps2*math.Abs(d.Get(m-1)))*math.Abs(d.Get(m-1-1))+safmin {
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
				Dlaev2(d.GetPtr(l-1-1), e.GetPtr(l-1-1), d.GetPtr(l-1), &rt1, &rt2, &c, &s)
				work.Set(m-1, c)
				work.Set((*n)-1+m-1, s)
				Dlasr('R', 'V', 'F', n, func() *int { y := 2; return &y }(), work.Off(m-1), work.Off((*n)-1+m-1), z.Off(0, l-1-1), ldz)
			} else {
				Dlae2(d.GetPtr(l-1-1), e.GetPtr(l-1-1), d.GetPtr(l-1), &rt1, &rt2)
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
		r = Dlapy2(&g, &one)
		g = d.Get(m-1) - p + (e.Get(l-1-1) / (g + signf64(r, g)))

		s = one
		c = one
		p = zero

		//        Inner loop
		lm1 = l - 1
		for i = m; i <= lm1; i++ {
			f = s * e.Get(i-1)
			b = c * e.Get(i-1)
			Dlartg(&g, &f, &c, &s, &r)
			if i != m {
				e.Set(i-1-1, r)
			}
			g = d.Get(i-1) - p
			r = (d.Get(i+1-1)-g)*s + two*c*b
			p = s * r
			d.Set(i-1, g+p)
			g = c*r - b

			//           If eigenvectors are desired, then save rotations.
			if icompz > 0 {
				work.Set(i-1, c)
				work.Set((*n)-1+i-1, s)
			}

		}

		//        If eigenvectors are desired, then apply saved rotations.
		if icompz > 0 {
			mm = l - m + 1
			Dlasr('R', 'V', 'F', n, &mm, work.Off(m-1), work.Off((*n)-1+m-1), z.Off(0, m-1), ldz)
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
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ssfmax, &anorm, toPtr(lendsv-lsv+1), func() *int { y := 1; return &y }(), d.MatrixOff(lsv-1, *n, opts), n, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ssfmax, &anorm, toPtr(lendsv-lsv), func() *int { y := 1; return &y }(), e.MatrixOff(lsv-1, *n, opts), n, info)
	} else if iscale == 2 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ssfmin, &anorm, toPtr(lendsv-lsv+1), func() *int { y := 1; return &y }(), d.MatrixOff(lsv-1, *n, opts), n, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &ssfmin, &anorm, toPtr(lendsv-lsv), func() *int { y := 1; return &y }(), e.MatrixOff(lsv-1, *n, opts), n, info)
	}

	//     Check for no convergence to an eigenvalue after a total
	//     of N*MAXIT iterations.
	if jtot < nmaxit {
		goto label10
	}
	for i = 1; i <= (*n)-1; i++ {
		if e.Get(i-1) != zero {
			(*info) = (*info) + 1
		}
	}
	return

	//     Order eigenvalues and eigenvectors.
label160:
	;
	if icompz == 0 {
		//        Use Quick Sort
		Dlasrt('I', n, d, info)

	} else {
		//        Use Selection Sort to minimize swaps of eigenvectors
		for ii = 2; ii <= (*n); ii++ {
			i = ii - 1
			k = i
			p = d.Get(i - 1)
			for j = ii; j <= (*n); j++ {
				if d.Get(j-1) < p {
					k = j
					p = d.Get(j - 1)
				}
			}
			if k != i {
				d.Set(k-1, d.Get(i-1))
				d.Set(i-1, p)
				goblas.Dswap(*n, z.Vector(0, i-1), 1, z.Vector(0, k-1), 1)
			}
		}
	}
}
