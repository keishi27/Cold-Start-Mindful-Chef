class NotFoundException(Exception):
	def __init__(self, message):
		self.message = message

class AuthenticationException(Exception):
	def __init__(self, message):
		self.message = message

class UnauthorizedException(Exception):
	def __init__(self, message):
		self.message = message

class BadRequestException(Exception):
	def __init__(self, message):
		self.message = message